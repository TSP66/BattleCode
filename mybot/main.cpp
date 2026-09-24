// Battlecode bot: the self-play policy, run forward once per turn.
//
// The judge meters wasm instructions, and its virtual clock is denominated in
// exactly those points, so the bot can read its own meter mid-turn. Two
// things follow, and both are used here:
//
//   * the trunk is run block by block with a check in between, and if another
//     block would not fit the bot abandons the network and answers with a
//     cheap legal move instead, so it can never be killed for overrunning;
//   * the first few turns report where the points actually went, which is how
//     the architecture gets sized against measurement rather than guesswork.
#include "helper.hpp"
#include "memory.hpp"
#include "net.hpp"
#include "obs.hpp"
#include "weights_data.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <cstdint>
#include <vector>

namespace {

constexpr std::int64_t TURN_BUDGET = 100'000'000;
// what must stay unspent: one metered write (2.5M + 4000/byte) plus room for
// the tail of the turn
constexpr std::int64_t RESERVE = 12'000'000;
// the 1x1 projection, the scalar branch and the two fused layers. The scalar
// branch is 708 inputs wide with the remembered features, not 14, which is
// ~0.3M more points.
constexpr std::int64_t HEAD_RESERVE = 7'000'000;
constexpr int LOG_UNTIL_ROUND = 3;

std::int64_t points_now() {
    auto const t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

net::Weights W;
bool have_net = false;
char const* net_status = "not loaded";

// One process is one dragon (helper.hpp reads its ID once, then loops over its
// turns), so one memory serves the whole life of this dragon and no other.
memory::Memory MEM;
float scalars[memory::N_SCALARS_IN];

std::vector<float> buf_local, buf_x, buf_y, buf_z, buf_col, buf_pad, buf_flat;
std::vector<float> buf_sc, buf_h, buf_fuse;
float logits[obs::N_ACTIONS];
std::uint8_t mask[obs::N_ACTIONS];

struct Layers {
    float const *stem_c, *stem_nw, *stem_nb;
    struct Block { float const *c1, *n1w, *n1b, *c2, *n2w, *n2b; };
    std::vector<Block> blocks;
    float const *flat_c, *flat_nw, *flat_nb;
    float const *sc0w, *sc0b, *sc2w, *sc2b;
    float const *fu0w, *fu0b, *fu2w, *fu2b, *piw, *pib;
} L;

void bind_layers() {
    int const w = W.width;
    W.at = 0;
    L.stem_c = W.take((std::size_t)w * 23 * 9);
    L.stem_nw = W.take((std::size_t)w);
    L.stem_nb = W.take((std::size_t)w);
    L.blocks.resize((std::size_t)W.blocks);
    for (auto& b : L.blocks) {
        b.c1 = W.take((std::size_t)w * w * 9);
        b.n1w = W.take((std::size_t)w);
        b.n1b = W.take((std::size_t)w);
        b.c2 = W.take((std::size_t)w * w * 9);
        b.n2w = W.take((std::size_t)w);
        b.n2b = W.take((std::size_t)w);
    }
    L.flat_c = W.take((std::size_t)W.head * w);
    L.flat_nw = W.take((std::size_t)W.head);
    L.flat_nb = W.take((std::size_t)W.head);
    L.sc0w = W.take((std::size_t)128 * (std::size_t)W.scalars);
    L.sc0b = W.take(128);
    L.sc2w = W.take(128 * 128);
    L.sc2b = W.take(128);
    int const fuse_in = W.head * obs::CELLS + 128;
    L.fu0w = W.take((std::size_t)W.hidden * fuse_in);
    L.fu0b = W.take((std::size_t)W.hidden);
    L.fu2w = W.take((std::size_t)W.hidden * W.hidden);
    L.fu2b = W.take((std::size_t)W.hidden);
    L.piw = W.take((std::size_t)obs::N_ACTIONS * W.hidden);
    L.pib = W.take(obs::N_ACTIONS);

    int const w_pitch = w * net::PITCH;
    buf_local.assign((std::size_t)obs::N_CHANNELS * net::PITCH, 0.0f);
    buf_x.assign((std::size_t)w_pitch, 0.0f);
    buf_y.assign((std::size_t)w_pitch, 0.0f);
    buf_z.assign((std::size_t)w_pitch, 0.0f);
    buf_col.assign((std::size_t)w * 9 * net::PITCH, 0.0f);
    buf_pad.assign((std::size_t)w * net::PAD_W * net::PAD_W, 0.0f);
    buf_flat.assign((std::size_t)W.head * net::PITCH, 0.0f);
    buf_sc.assign(256, 0.0f);
    buf_h.assign((std::size_t)fuse_in, 0.0f);
    buf_fuse.assign((std::size_t)W.hidden * 2, 0.0f);
}

// Runs the policy. Returns false when the budget ran out mid-trunk, in which
// case `logits` is meaningless and the caller falls back.
bool forward(obs::Snapshot const& snap, std::int64_t start, std::int64_t* stage_cost) {
    int const w = W.width;

    snap.local(buf_local.data());
    // the observation is written 49 cells to a channel; spread it onto the
    // padded pitch the kernels use
    for (int c = obs::N_CHANNELS - 1; c >= 0; c--) {
        float* dst = buf_local.data() + (std::size_t)c * net::PITCH;
        float const* src = buf_local.data() + (std::size_t)c * obs::CELLS;
        std::memmove(dst, src, sizeof(float) * obs::CELLS);
        dst[49] = dst[50] = dst[51] = 0.0f;
    }
    stage_cost[0] = points_now() - start;

    net::im2col(buf_local.data(), obs::N_CHANNELS, buf_pad.data(), buf_col.data());
    net::gemm(L.stem_c, buf_col.data(), buf_x.data(), w, obs::N_CHANNELS * 9);
    net::group_norm(buf_x.data(), w, L.stem_nw, L.stem_nb);
    net::apply_silu(buf_x.data(), w * net::PITCH);
    stage_cost[1] = points_now() - start;

    // Before the first block there is nothing measured to extrapolate from
    // but the stem, so scale it by the ratio of the work: a block is two
    // width->width convolutions against the stem's single 23->width one.
    std::int64_t est_block = (stage_cost[1] - stage_cost[0]) * 2 * w / obs::N_CHANNELS;
    int done = 0;
    for (auto const& b : L.blocks) {
        std::int64_t const before = points_now();
        // refuse to start a block that would not leave room for the head and
        // for answering the turn
        if (before - start + est_block * 5 / 4 + HEAD_RESERVE > TURN_BUDGET - RESERVE) break;

        net::im2col(buf_x.data(), w, buf_pad.data(), buf_col.data());
        net::gemm(b.c1, buf_col.data(), buf_y.data(), w, w * 9);
        net::group_norm(buf_y.data(), w, b.n1w, b.n1b);
        net::apply_silu(buf_y.data(), w * net::PITCH);

        net::im2col(buf_y.data(), w, buf_pad.data(), buf_col.data());
        net::gemm(b.c2, buf_col.data(), buf_z.data(), w, w * 9);
        net::group_norm(buf_z.data(), w, b.n2w, b.n2b);
        for (int i = 0; i < w * net::PITCH; i++)
            buf_x[(std::size_t)i] = net::silu(buf_x[(std::size_t)i] + buf_z[(std::size_t)i]);

        est_block = points_now() - before;
        done++;
    }
    stage_cost[2] = points_now() - start;
    stage_cost[4] = done;
    if (done < W.blocks) return false;

    // 1x1 projection: a gemm straight over the channel axis, no im2col
    net::gemm(L.flat_c, buf_x.data(), buf_flat.data(), W.head, w);
    net::group_norm(buf_flat.data(), W.head, L.flat_nw, L.flat_nb);
    net::apply_silu(buf_flat.data(), W.head * net::PITCH);
    for (int c = 0; c < W.head; c++)
        std::memcpy(buf_h.data() + (std::size_t)c * obs::CELLS,
                    buf_flat.data() + (std::size_t)c * net::PITCH,
                    sizeof(float) * obs::CELLS);

    net::linear(L.sc0w, L.sc0b, scalars, buf_sc.data(), 128, W.scalars);
    net::apply_silu(buf_sc.data(), 128);
    net::linear(L.sc2w, L.sc2b, buf_sc.data(), buf_sc.data() + 128, 128, 128);
    net::apply_silu(buf_sc.data() + 128, 128);
    std::memcpy(buf_h.data() + (std::size_t)W.head * obs::CELLS,
                buf_sc.data() + 128, sizeof(float) * 128);

    int const fuse_in = W.head * obs::CELLS + 128;
    net::linear(L.fu0w, L.fu0b, buf_h.data(), buf_fuse.data(), W.hidden, fuse_in);
    net::apply_silu(buf_fuse.data(), W.hidden);
    float* second = buf_fuse.data() + W.hidden;
    net::linear(L.fu2w, L.fu2b, buf_fuse.data(), second, W.hidden, W.hidden);
    net::apply_silu(second, W.hidden);
    net::linear(L.piw, L.pib, second, logits, obs::N_ACTIONS, W.hidden);
    stage_cost[3] = points_now() - start;
    return true;
}

// Cheap legal move for the first turn, for a budget bail-out, and for when
// the weights are missing: take a pearl if one is a step away, else go
// straight on, else anything legal -- but never walk onto another head.
//
// That last rule is not a nicety. Entering a head's cell is legal, and it
// kills both dragons, so a blind step into one is a trade at best. Every
// dragon's first turn runs this heuristic (the turn that widens the weights
// cannot also afford a forward pass), and a split child's first turn is
// exactly when a friendly head is most likely to be one step away: on
// Default all four dragons split on round 1 and each child's blind "straight
// on" took out an ally, six of our eight dragons gone before round 2. It cost
// a server game 0-12 in two rounds against an opponent that did not
// reciprocate. v10 and every submission before it did the same.
int fallback(obs::Snapshot const& snap) {
    int any = -1, safe = -1;
    std::array<int, 3> dirs{};
    for (int id = 0; id < obs::N_MOVES; id++) {
        if (!mask[id]) continue;
        if (any < 0) any = id;
        if (obs::TABLES.n_steps[id] != 1) continue;
        obs::decode_move(id, snap.facing, dirs);
        int const nx = ((snap.hx + obs::DX[(std::size_t)dirs[0]]) % snap.w + snap.w) % snap.w;
        int const ny = ((snap.hy + obs::DY[(std::size_t)dirs[0]]) % snap.h + snap.h) % snap.h;
        int const cell = snap.cell_of(nx, ny);
        if (cell >= 0) {
            auto const* part = snap.tile(cell).get_dragon();
            if (part != nullptr && part->is_dragon_head) continue;   // a trade, not a move
            if (snap.tile(cell).pearl) return id;
        }
        if (safe < 0 || id == 0) safe = id;                          // prefer straight on
    }
    return safe >= 0 ? safe : (any < 0 ? 0 : any);
}

// The turn's whole answer, sent as one write.
//
// The judge charges every write syscall 2.5M points, and stdout is not
// buffered, so the helper's `std::cout << a << b` style costs a syscall per
// `<<`: MOVE plus ENDTURN came to ~14M a turn, and a debug LOG line to ~105M.
// Building the text first and writing it once costs 2.5M plus 4000 a byte.
std::string out;

void emit(int action, obs::Snapshot const& snap) {
    if (action < obs::N_MOVES) {
        static constexpr char NAMES[4] = {'N', 'E', 'S', 'W'};
        std::array<int, 3> dirs{};
        int const n = obs::decode_move(action, snap.facing, dirs);
        out += "MOVE ";
        for (int i = 0; i < n; i++) out += NAMES[dirs[(std::size_t)i]];
        out += '\n';
        return;
    }
    int const k = obs::SPLIT_K[(std::size_t)(action - obs::N_MOVES)];
    out += "SPLIT ";
    out += std::to_string(k < 0 ? snap.length / 2 : k);
    out += '\n';
}

// Sonar is sensing, not an action: it does not consume the turn, and because
// these lines join the one write the bot already makes they cost 0.51M points a
// turn in total -- 0.6% of the cap, measured with wasmprobe/meter_bot.sh.
//
// Broadcast in all four directions with a zero payload, which is exactly what
// the training environment casts (bc_vec.hpp), so the five echo counts the
// policy reads here are the ones it was trained on. The payload stays 0 until
// the memory codec is wired up; 0 also fits 32 bits, so it is never dropped for
// a receiver still on the legacy protocol.
//
// PROTOCOL must be declared every turn, not once: without it the engine keeps
// this dragon on the legacy protocol and sends no ECHOES line at all.
// Only a net trained with sonar broadcasts. This matters: our own rays land on
// our own dragons, so broadcasting makes num_msgs (scalar 12) non-zero, and a
// net trained without sonar saw that column as always zero. Switching it on
// under such a net would feed it an input it has never seen for no benefit, so
// a 14- or 708-scalar checkpoint keeps the exact legacy behaviour -- no
// PROTOCOL line, no SONAR lines, no ECHOES in the blocks it is sent.
constexpr bool USE_SONAR = embedded::SCALARS == memory::N_SCALARS_IN;

void flush_turn() {
    if (USE_SONAR) {
        for (char d : {'N', 'E', 'S', 'W'}) {
            out += "SONAR ";
            out += d;
            out += " 0\n";
        }
        out += "PROTOCOL 3\n";
    }
    out += "ENDTURN\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
    out.clear();
}

}  // namespace

// The weights are compiled in (weights_data.hpp). They used to travel as a
// data file, but the judge runs the bot as wasm with no filesystem, so the
// file was never found and every turn silently used the fallback. Nothing
// here touches the filesystem now.
bool load_weights() {
    if (!net::load_embedded(embedded::HI, embedded::LO, embedded::COUNT, embedded::CHECKSUM,
                            embedded::WIDTH, embedded::BLOCKS, embedded::HIDDEN,
                            embedded::HEAD, embedded::SCALARS, memory::N_SCALARS_IN,
                            W, &net_status))
        return false;
    bind_layers();
    // every parameter bound exactly once, or the layout disagrees with the
    // exporter and the net must not run
    if (W.at != W.data.size()) {
        net_status = "layout does not consume the blob";
        return false;
    }
    return true;
}

int main() {
    // one full-buffered stream, flushed once a turn by flush_turn()
    static char stdout_buf[1 << 12];
    std::setvbuf(stdout, stdout_buf, _IOFBF, sizeof(stdout_buf));
    auto [ct, game] = unswbc::init();
    have_net = load_weights();
    MEM.init(game.width, game.height);

    obs::Snapshot snap;
    bool first = true;

    while (unswbc::update(ct, game)) {
        std::int64_t const start = points_now();
        snap.build(ct, game);
        snap.mask(ct, game, mask);
        // The memory is updated on every turn, including the first and any
        // turn the network does not get to run on: a gap in it would change
        // every feature after it. Building the scalars here rather than inside
        // forward() keeps that order -- observe this turn, then read it back --
        // the one clone_features.py used.
        MEM.observe(snap, game.round_num);
        snap.scalars(ct, game, scalars);
        MEM.features(snap, game.round_num, scalars + obs::N_SCALARS);
        obs::echo_scalars(ct, scalars + obs::N_SCALARS + memory::N_EXTRA);
        std::int64_t const t_obs = points_now() - start;

        int action = -1;
        std::int64_t stage[5]{};
        bool ran = false;
        // the first turn already paid to widen ~1.9M weights, so skip the
        // network rather than risk stacking a full forward pass on top
        if (have_net && !first) {
            ran = forward(snap, start, stage);
            if (ran) {
                float best = -1e30f;
                for (int i = 0; i < obs::N_ACTIONS; i++)
                    if (mask[i] && logits[i] > best) { best = logits[i]; action = i; }
            }
        }
        if (action < 0) action = fallback(snap);
#ifdef BC_DUMP
        // parity testing only (wasmprobe/parity_net.py): the observation and
        // logits this turn, on stderr, which the judge never sees
        {
            std::vector<float> loc((std::size_t)obs::N_CHANNELS * obs::CELLS);
            snap.local(loc.data());
            std::fprintf(stderr, "DUMP %d %d", ran ? 1 : 0, action);
            for (float v : loc) std::fprintf(stderr, " %.6g", v);
            // every scalar the bot can feed, the remembered ones included, so
            // the parity checks see exactly what the network was given
            for (float v : scalars) std::fprintf(stderr, " %.9g", v);
            for (int i = 0; i < obs::N_ACTIONS; i++) std::fprintf(stderr, " %d", (int)mask[i]);
            for (int i = 0; i < obs::N_ACTIONS; i++) std::fprintf(stderr, " %.6g", ran ? logits[i] : 0.0f);
            std::fprintf(stderr, "\n");
        }
#endif
        // Shown in the replay viewer, so a replay tells whether the network
        // drove each turn. The silent fallback hid a missing net for two
        // submissions; this is ~60k points a turn.
        out += ran ? "INDICATOR net\n"
                   : (have_net ? (first ? "INDICATOR first-turn\n" : "INDICATOR budget\n")
                               : "INDICATOR NO NET\n");

#ifdef BC_LOG
        // stage costs, for metering builds; the server drops LOG from replays
        if (game.round_num <= LOG_UNTIL_ROUND) {
            int legal = 0;
            for (int i = 0; i < obs::N_ACTIONS; i++) legal += mask[i];
            out += "LOG pts obs " + std::to_string(t_obs) +
                   " stem " + std::to_string(stage[1]) +
                   " trunk " + std::to_string(stage[2]) +
                   " head " + std::to_string(stage[3]) +
                   " blocks " + std::to_string(stage[4]) + " of " + std::to_string(W.blocks) +
                   " net " + std::to_string(have_net ? 1 : 0) + " (" + net_status + ")" +
                   " legal " + std::to_string(legal) + " act " + std::to_string(action) +
                   " total " + std::to_string(points_now() - start) + "\n";
        }
#else
        (void)t_obs;
#endif
        emit(action, snap);
        flush_turn();
        first = false;
    }
}
