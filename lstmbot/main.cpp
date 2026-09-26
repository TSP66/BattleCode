// The LSTM policy bot (bcsim/train/lstm_net.py), one process per dragon.
//
// Each turn: parse the block; update this dragon's world-anchored memory from
// its 7x7 window and its own body from what it can see; render the 38 x 14 x 14
// grid in its own frame exactly as the simulator does (cpp/bc_memory.hpp
// grid_cfg + bc_obs.hpp); run the CNN + LSTM (lstmnet.hpp, weights from
// bcsim/train/export_lstm.py) carrying h/c and the previous action between
// turns; play the masked argmax; broadcast four sonars and declare protocol 3,
// as training did (bcsim sonar=True).
//
// -DBC_DUMP (native parity builds only) prints, per turn on stderr: the previous
// action, the grid as Q12 integers, the mask, the logits and the action.
#include "helper.hpp"
#include "lstmnet.hpp"
#include "obs.hpp"
#include "weights.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace {

using ln::i16;
using ln::i32;

constexpr int G = 14, GC = G * G;          // grid side, cells (196, a multiple of 4)
constexpr int HALF = 7;                    // head at row 7, col 7
constexpr int D = 7, DC = 52;              // after the stride-2 conv: 7x7, padded to 52
constexpr int NCH = 38;
constexpr int N_ACT = obs::N_ACTIONS;      // 48
constexpr int NO_ACTION = N_ACT;
constexpr int PREV_DIM = 48;
constexpr int MAX_ROUNDS = 500;            // the simulator's cfg max_rounds, and the judge's
constexpr int XIN = wt::EMB + PREV_DIM;    // 304, what LayerNorm and LSTM layer 0 read
constexpr i16 ONE = 4096;                  // Q12

// grid channels (bcsim/cpp/bc_memory.hpp grid_cfg, train/lstm_net.py CHANNELS)
enum : int {
    KELP = 0, PORTAL = 4, NEVER_SPAWNS = 8,
    PEARL = 9, PEARL_TIMER = 10, ALLY_HEAD = 11, ALLY_BODY = 12, ENEMY_HEAD = 13,
    ENEMY_BODY = 14, SEG_DIR = 15,
    SELF_BODY = 19, SELF_INDEX = 20, SELF_TAIL = 21,
    SEEN = 22, PEARL_EXPECTED = 23, PEARL_TIMER_PROJ = 24, ENEMY_MEM = 25, ALLY_MEM = 26,
    VISITED = 27, ROUND = 28, LENGTH = 29, UNITS = 30, MAP_W = 31, MAP_H = 32, ECHO = 33,
};

// the judge's virtual clock counts CPU points, so this is the bot's own meter
std::int64_t points_now() {
    auto const t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

// ---------------------------------------------------------------- weights
std::vector<i16> WB;
std::size_t at = 0, fat = 0;
i16 const* take(std::size_t n) { i16 const* p = WB.data() + at; at += n; return p; }
float const* ftake(std::size_t n) { float const* p = wt::FP + fat; fat += n; return p; }

struct Layer { i16 const* w; float const *s, *b, *g = nullptr, *beta = nullptr; int cin, cout, k; };
Layer conv(int cin, int cout, int k, bool norm) {
    Layer l{};
    l.cin = cin; l.cout = cout; l.k = k;
    l.w = take((std::size_t)cout * cin * k * k);
    l.s = ftake((std::size_t)cout);
    l.b = ftake((std::size_t)cout);
    if (norm) { l.g = ftake((std::size_t)cout); l.beta = ftake((std::size_t)cout); }
    return l;
}
Layer lin(int in, int out) {
    Layer l{};
    l.cin = in; l.cout = out; l.k = 1;
    l.w = take((std::size_t)in * out);
    l.s = ftake((std::size_t)out);
    l.b = ftake((std::size_t)out);
    return l;
}

Layer stem, down, squeeze, embed, pi;
std::vector<Layer> res15, res8, lstm;
float const *prev_table, *ln_g, *ln_b;

bool load() {
    ln::build_silu();
    WB.resize(wt::COUNT + 16);
    ln::join_planes(wt::HI, wt::LO, wt::COUNT, WB.data());
    stem = conv(NCH, wt::C1, 3, true);
    for (int i = 0; i < 2 * wt::B1; i++) res15.push_back(conv(wt::C1, wt::C1, 3, true));
    down = conv(wt::C1, wt::C2, 3, true);
    for (int i = 0; i < 2 * wt::B2; i++) res8.push_back(conv(wt::C2, wt::C2, 3, true));
    squeeze = conv(wt::C2, wt::SQ, 1, false);
    embed = lin(wt::SQ * D * D, wt::EMB);
    for (int l = 0; l < wt::LAYERS; l++)
        lstm.push_back(lin((l == 0 ? XIN : wt::HID) + wt::HID, 4 * wt::HID));
    pi = lin(wt::HID, N_ACT);
    prev_table = ftake((std::size_t)(N_ACT + 1) * PREV_DIM);
    ln_g = ftake(XIN);
    ln_b = ftake(XIN);
    return at == wt::COUNT && fat == wt::N_FP;
}

// ---------------------------------------------------------------- memory
// One cell of the world as this dragon last saw it (bc_memory.hpp DragonMemory).
constexpr int NEVER = -10000;
struct Cell {
    std::int16_t seen = NEVER, visit = NEVER, foe = NEVER, ally = NEVER;
    std::int8_t cd = -1;
    std::uint8_t pearl = 0;
    std::uint8_t sides = 0;                  // bits 0-3 kelp on world side N E S W, 4-7 portal
};
std::vector<Cell> world;
int W_ = 0, H_ = 0;

// GridDecay: round(4096 * exp(-age / tau)), 0 from age 256
i16 DEC32[256], DEC64[256], DEC8[256], DEC16[256];
void decay_tables() {
    for (int a = 0; a < 256; a++) {
        DEC32[a] = (i16)std::nearbyint(4096.0 * std::exp(-a / 32.0));
        DEC64[a] = (i16)std::nearbyint(4096.0 * std::exp(-a / 64.0));
        DEC8[a] = (i16)std::nearbyint(4096.0 * std::exp(-a / 8.0));
        DEC16[a] = (i16)std::nearbyint(4096.0 * std::exp(-a / 16.0));
    }
}
inline i16 decay(i16 const* t, int age) { return (unsigned)age < 256u ? t[age] : 0; }

// ego offset -> world offset, per facing (N, E, S, W): bc_obs.hpp ego_to_world
inline void ego_to_world(int f, int ox, int oy, int& wx, int& wy) {
    switch (f) {
        case 0: wx = ox; wy = oy; break;
        case 1: wx = -oy; wy = ox; break;
        case 2: wx = -ox; wy = -oy; break;
        default: wx = oy; wy = -ox; break;
    }
}
inline void world_to_ego(int f, int wx, int wy, int& ox, int& oy) {
    switch (f) {
        case 0: ox = wx; oy = wy; break;
        case 1: ox = wy; oy = -wx; break;
        case 2: ox = -wx; oy = -wy; break;
        default: ox = -wy; oy = wx; break;
    }
}
inline int wrap(int v, int m) { return (v % m + m) % m; }
inline Cell& cell_at(int x, int y) { return world[(std::size_t)wrap(y, H_) * W_ + wrap(x, W_)]; }

float loc[obs::N_CHANNELS * obs::CELLS];

// The window into memory, in the order bc_obs.hpp writes it (see(), sides, foe/ally).
void observe(obs::Snapshot const& s, int round) {
    for (int i = 0; i < obs::CELLS; i++) {
        auto const& t = s.tile(i);
        Cell& m = cell_at(t.position.x, t.position.y);
        m.seen = (std::int16_t)round;
        m.cd = t.pearl_time < 0 ? -1 : (std::int8_t)(t.pearl_time < 99 ? t.pearl_time : 99);
        m.pearl = t.pearl ? 1 : 0;
        std::uint8_t sides = 0;
        for (int d = 0; d < 4; d++) {
            auto const kind = t.edges[(std::size_t)d].edge_type;
            if (kind == unswbc::EdgeType::KELP) sides |= (std::uint8_t)(1 << d);
            else if (kind == unswbc::EdgeType::PORTAL) sides |= (std::uint8_t)(1 << (4 + d));
        }
        m.sides = sides;
        if (auto const* part = t.get_dragon()) {
            if (part->dragon_id == s.my_id) continue;
            if ((char)part->team.value == s.my_team) m.ally = (std::int16_t)round;
            else m.foe = (std::int16_t)round;
        }
    }
    cell_at(s.hx, s.hy).visit = (std::int16_t)round;
}

// ---------------------------------------------------------------- own body
// The simulator draws the whole body into the grid; the window shows at most a
// few segments. So the bot tracks it: this turn's visible chain, then the rest
// of last turn's body, shifted by however many steps the head moved -- worked
// out by matching the chain against last turn's body, not from the action, so a
// portal or a replayed transcript cannot desynchronise it.
std::deque<std::pair<int, int>> body;       // world cells, head first

void track_body(obs::Snapshot const& s) {
    std::vector<std::pair<int, int>> vis;
    for (int i = 0; i < s.chain_len(); i++) {
        auto const& p = s.tile(s.chain_cell(i)).position;
        vis.emplace_back(p.x, p.y);
    }
    if (vis.empty()) vis.emplace_back(s.hx, s.hy);
    std::deque<std::pair<int, int>> nb(vis.begin(), vis.end());
    int const v = (int)vis.size();
    if (!body.empty()) {
        // shift n: vis[n + j] == old[j] for the overlap
        int shift = -1;
        for (int n = 0; n <= v && shift < 0; n++) {
            bool ok = n < v;
            for (int j = 0; n + j < v && j < (int)body.size(); j++)
                if (vis[(std::size_t)(n + j)] != body[(std::size_t)j]) { ok = false; break; }
            if (ok) shift = n;
        }
        if (shift >= 0)
            for (std::size_t j = (std::size_t)(v - shift); j < body.size(); j++) nb.push_back(body[j]);
    }
    // a body we have never seen the end of (a dragon's first turns, a split
    // child): continue straight on from the last known link
    while ((int)nb.size() < s.length) {
        std::pair<int, int> const last = nb.back();
        int dx = 0, dy = 0;
        if (nb.size() >= 2) {
            auto const prev = nb[nb.size() - 2];
            dx = last.first - prev.first;
            dy = last.second - prev.second;
            if (dx > 1) dx = -1;
            if (dx < -1) dx = 1;
            if (dy > 1) dy = -1;
            if (dy < -1) dy = 1;
        } else {
            dx = -obs::DX[(std::size_t)s.facing];
            dy = -obs::DY[(std::size_t)s.facing];
        }
        nb.emplace_back(wrap(last.first + dx, W_), wrap(last.second + dy, H_));
    }
    while ((int)nb.size() > s.length) nb.pop_back();
    body = std::move(nb);
}

// ---------------------------------------------------------------- the grid
alignas(16) i16 grid[NCH * GC];

void render(obs::Snapshot const& s, int round, float units, float const* echo) {
    std::memset(grid, 0, sizeof(grid));
    auto at_ = [&](int ch, int cell) -> i16& { return grid[ch * GC + cell]; };
    for (int r = 0; r < G; r++)
        for (int c = 0; c < G; c++) {
            int const cell = r * G + c;
            int wx, wy;
            ego_to_world(s.facing, c - HALF, r - HALF, wx, wy);
            Cell const& m = cell_at(s.hx + wx, s.hy + wy);
            if (m.visit > NEVER) at_(VISITED, cell) = decay(DEC16, round - m.visit);
            if (m.seen <= NEVER) continue;
            int const age = round - m.seen;
            for (int d = 0; d < 4; d++) {
                int const wd = (d + s.facing) & 3;
                if (m.sides >> wd & 1) at_(KELP + d, cell) = ONE;
                if (m.sides >> (4 + wd) & 1) at_(PORTAL + d, cell) = ONE;
            }
            if (m.cd < 0) at_(NEVER_SPAWNS, cell) = ONE;
            at_(SEEN, cell) = decay(DEC32, age);
            bool const expect = m.pearl || (m.cd >= 0 && m.cd <= age);
            if (expect) at_(PEARL_EXPECTED, cell) = decay(DEC64, age);
            if (m.cd >= 0) at_(PEARL_TIMER_PROJ, cell) = ln::to_q((float)(m.cd > age ? m.cd - age : 0) / 99.0f, ln::QG);
            if (m.foe > NEVER) at_(ENEMY_MEM, cell) = decay(DEC8, round - m.foe);
            if (m.ally > NEVER) at_(ALLY_MEM, cell) = decay(DEC8, round - m.ally);
        }
    // live, from the observation the flat bot already builds (it matches the simulator's)
    static constexpr int LIVE[10][2] = {
        {obs::C_PEARL, PEARL}, {obs::C_PEARL_TIME, PEARL_TIMER},
        {obs::C_ALLY_HEAD, ALLY_HEAD}, {obs::C_ALLY_BODY, ALLY_BODY},
        {obs::C_ENEMY_HEAD, ENEMY_HEAD}, {obs::C_ENEMY_BODY, ENEMY_BODY},
        {obs::C_FACE, SEG_DIR}, {obs::C_FACE + 1, SEG_DIR + 1},
        {obs::C_FACE + 2, SEG_DIR + 2}, {obs::C_FACE + 3, SEG_DIR + 3}};
    for (int r = 0; r < obs::WINDOW; r++)
        for (int c = 0; c < obs::WINDOW; c++) {
            int const i = r * obs::WINDOW + c, cell = (r + HALF - obs::VISION) * G + (c + HALF - obs::VISION);
            for (auto const& lv : LIVE) at_(lv[1], cell) = ln::to_q(loc[lv[0] * obs::CELLS + i], ln::QG);
        }
    // self: the tracked body, head excluded, as bc_obs.hpp draws the true one
    int const len = (int)body.size();
    for (int i = 1; i < len; i++) {
        int ddx = body[(std::size_t)i].first - s.hx, ddy = body[(std::size_t)i].second - s.hy;
        if (ddx > W_ / 2) ddx -= W_;
        if (ddx < -(W_ - 1) / 2) ddx += W_;
        if (ddy > H_ / 2) ddy -= H_;
        if (ddy < -(H_ - 1) / 2) ddy += H_;
        int ox, oy;
        world_to_ego(s.facing, ddx, ddy, ox, oy);
        if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
        int const cell = (oy + HALF) * G + (ox + HALF);
        at_(SELF_BODY, cell) = ONE;
        at_(SELF_INDEX, cell) = ln::to_q((float)i / (float)(len - 1 > 1 ? len - 1 : 1), ln::QG);
        if (i == len - 1) at_(SELF_TAIL, cell) = ONE;
    }
    float const glob[10] = {(float)round / MAX_ROUNDS, (float)(s.length < 64 ? s.length : 64) / 64.0f,
                            units, (float)W_ / 64.0f, (float)H_ / 64.0f,
                            echo[0], echo[1], echo[2], echo[3], echo[4]};
    for (int k = 0; k < 10; k++) {
        i16 const q = ln::to_q(glob[k], ln::QG);
        i16* p = grid + (ROUND + k) * GC;
        for (int j = 0; j < GC; j++) p[j] = q;
    }
}

// ---------------------------------------------------------------- forward
constexpr int MAXC = wt::C1 > wt::C2 ? wt::C1 : wt::C2;
alignas(16) i16 act_a[MAXC * GC], act_b[MAXC * GC], act_c[MAXC * GC];
alignas(16) i16 col[(NCH > MAXC ? NCH : MAXC) * 9 * GC + 64];
alignas(16) i16 padbuf[(G + 2) * (G + 2)];
alignas(16) i16 pads[(NCH > MAXC ? NCH : MAXC) * (G + 2) * (G + 2) + 32];
// im2col_s1 writes up to 7 cells past a row; slack so the last pair cannot run off `col`
alignas(16) i32 acc[4 * wt::HID > MAXC * GC ? 4 * wt::HID : MAXC * GC];
float zbuf[MAXC * GC];
alignas(16) i16 flat[wt::SQ * D * D + 8];
alignas(16) i16 xin[XIN + 2 * wt::HID + 8];
float xf[XIN], zg[4 * wt::HID];
float cstate[wt::LAYERS][wt::HID], hstate[wt::LAYERS][wt::HID];
alignas(16) i16 hq[wt::LAYERS][wt::HID];
int prev_action = NO_ACTION;

#ifdef BC_METER
std::int64_t m_gemm = 0, m_im2col = 0, m_post = 0;
#endif
void conv3(Layer const& L, i16 const* in, float qin, int h, int w, int Pin, int stride, int ho, int wo,
           int Pout, i16 const* skip, i16* out) {
#ifdef BC_METER
    std::int64_t const a0 = points_now();
#endif
#if LN_SIMD
    if (stride == 1) ln::im2col_s1(in, L.cin, h, w, Pin, Pout, pads, col);
    else ln::im2col(in, L.cin, h, w, Pin, stride, ho, wo, Pout, padbuf, col);
#else
    ln::im2col(in, L.cin, h, w, Pin, stride, ho, wo, Pout, padbuf, col);
#endif
#ifdef BC_METER
    std::int64_t const a1 = points_now();
#endif
    ln::gemm(L.w, L.cout, L.cin * 9, col, Pout, acc);
#ifdef BC_METER
    std::int64_t const a2 = points_now();
#endif
    ln::dequant(acc, L.s, L.b, L.cout, Pout, ho * wo, qin, zbuf);
    ln::gn_silu(zbuf, L.cout, Pout, ho * wo, L.g, L.beta, skip, out);
#ifdef BC_METER
    m_im2col += a1 - a0; m_gemm += a2 - a1; m_post += points_now() - a2;
#endif
}

#ifdef BC_DUMP
int dump_layers = 1;                           // the first turn's intermediates, for parity debugging
void dump_q(char const* tag, i16 const* x, int C, int P, int ncell) {
    if (!dump_layers) return;
    std::fprintf(stderr, "LAYER %s", tag);
    for (int c = 0; c < C; c++)
        for (int j = 0; j < ncell; j++) std::fprintf(stderr, " %d", (int)x[c * P + j]);
    std::fprintf(stderr, "\n");
}
void dump_f(char const* tag, float const* x, int n) {
    if (!dump_layers) return;
    std::fprintf(stderr, "LAYER %s", tag);
    for (int i = 0; i < n; i++) std::fprintf(stderr, " %.6g", x[i]);
    std::fprintf(stderr, "\n");
}
#else
inline void dump_q(char const*, i16 const*, int, int, int) {}
inline void dump_f(char const*, float const*, int) {}
#endif

void forward(float* logits) {
    conv3(stem, grid, ln::QG, G, G, GC, 1, G, G, GC, nullptr, act_a);
    dump_q("stem", act_a, wt::C1, GC, GC);
    for (int i = 0; i < wt::B1; i++) {
        conv3(res15[(std::size_t)(2 * i)], act_a, ln::QA, G, G, GC, 1, G, G, GC, nullptr, act_b);
        conv3(res15[(std::size_t)(2 * i + 1)], act_b, ln::QA, G, G, GC, 1, G, G, GC, act_a, act_c);
        std::memcpy(act_a, act_c, sizeof(i16) * wt::C1 * GC);
    }
    dump_q("res15", act_a, wt::C1, GC, GC);
    conv3(down, act_a, ln::QA, G, G, GC, 2, D, D, DC, nullptr, act_b);
    dump_q("down", act_b, wt::C2, DC, D * D);
    for (int i = 0; i < wt::B2; i++) {
        conv3(res8[(std::size_t)(2 * i)], act_b, ln::QA, D, D, DC, 1, D, D, DC, nullptr, act_c);
        conv3(res8[(std::size_t)(2 * i + 1)], act_c, ln::QA, D, D, DC, 1, D, D, DC, act_b, act_a);
        std::memcpy(act_b, act_a, sizeof(i16) * wt::C2 * DC);
    }
    dump_q("res8", act_b, wt::C2, DC, D * D);
    // squeeze 1x1 -> SiLU -> flatten
    ln::interleave(act_b, wt::C2, DC, col);
    ln::gemm(squeeze.w, wt::SQ, wt::C2, col, DC, acc);
    ln::dequant(acc, squeeze.s, squeeze.b, wt::SQ, DC, D * D, ln::QA, zbuf);
    for (int c = 0; c < wt::SQ; c++)
        for (int j = 0; j < D * D; j++) flat[c * D * D + j] = ln::to_q(ln::silu(zbuf[c * DC + j]), ln::QA);
    // embed -> SiLU, the previous action, LayerNorm
    ln::gemm(flat, 1, wt::SQ * D * D, embed.w, wt::EMB, acc);
    ln::dequant_vec(acc, embed.s, embed.b, wt::EMB, zbuf);
    for (int i = 0; i < wt::EMB; i++) xf[i] = ln::silu(zbuf[i]);
    for (int i = 0; i < PREV_DIM; i++) xf[wt::EMB + i] = prev_table[prev_action * PREV_DIM + i];
    ln::layer_norm(xf, XIN, ln_g, ln_b);
    dump_q("flat", flat, 1, wt::SQ * D * D, wt::SQ * D * D);
    dump_f("ln", xf, XIN);
    for (int i = 0; i < XIN; i++) xin[i] = ln::to_q(xf[i], ln::QA);
    int in = XIN;
    for (int l = 0; l < wt::LAYERS; l++) {
        std::memcpy(xin + in, hq[l], sizeof(i16) * wt::HID);
        Layer const& L = lstm[(std::size_t)l];
        ln::gemm(xin, 1, in + wt::HID, L.w, 4 * wt::HID, acc);
        ln::dequant_vec(acc, L.s, L.b, 4 * wt::HID, zg);
        ln::lstm_cell(zg, wt::HID, cstate[l], hstate[l], hq[l]);
        std::memcpy(xin, hq[l], sizeof(i16) * wt::HID);
        in = wt::HID;
    }
    ln::gemm(xin, 1, wt::HID, pi.w, N_ACT, acc);
    ln::dequant_vec(acc, pi.s, pi.b, N_ACT, logits);
    dump_f("h_last", hstate[wt::LAYERS - 1], wt::HID);
#ifdef BC_DUMP
    dump_layers = 0;
#endif
}

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

// Four sonars with a zero payload and protocol 3, every turn: what the training
// simulator casts (bcsim sonar=True), so the echo planes read what they trained on.
void flush_turn() {
    for (char d : {'N', 'E', 'S', 'W'}) {
        out += "SONAR ";
        out += d;
        out += " 0\n";
    }
    out += "PROTOCOL 3\nENDTURN\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
    out.clear();
}

}  // namespace

int main() {
    static char stdout_buf[1 << 12];
    std::setvbuf(stdout, stdout_buf, _IOFBF, sizeof(stdout_buf));
#ifdef BC_METER
    std::int64_t const t_start = points_now();
#endif
    auto [ct, game] = unswbc::init();
    bool const ok = load();
#ifdef BC_METER
    std::fprintf(stderr, "METER startup %lld\n", (long long)(points_now() - t_start));
#endif
    decay_tables();
    W_ = game.width;
    H_ = game.height;
    world.assign((std::size_t)W_ * H_, Cell{});

    obs::Snapshot snap;
    std::uint8_t mask[N_ACT];
    float logits[N_ACT];
    while (unswbc::update(ct, game)) {
#ifdef BC_METER
        std::int64_t const t0 = points_now();
#endif
        snap.build(ct, game);
        snap.mask(ct, game, mask);
        snap.local(loc);
        float echo[obs::N_ECHO];
        obs::echo_scalars(ct, echo);
        observe(snap, game.round_num);
        track_body(snap);
        render(snap, game.round_num, (float)ct.get_unit_count() / (float)game.unit_limit, echo);
#ifdef BC_METER
        std::int64_t const t1 = points_now();
#endif
        int action = -1;
        if (ok) {
            forward(logits);
#ifdef BC_METER
            std::fprintf(stderr, "METER obs+render %lld forward %lld (convs: im2col %lld gemm %lld dequant+gn+silu %lld)\n",
                         (long long)(t1 - t0), (long long)(points_now() - t1), (long long)m_im2col,
                         (long long)m_gemm, (long long)m_post);
            m_im2col = m_gemm = m_post = 0;
#endif
            float best = -3.4e38f;
            for (int a = 0; a < N_ACT; a++)
                if (mask[a] && logits[a] > best) { best = logits[a]; action = a; }
        }
        if (action < 0)                                       // no net, or no legal move
            for (int a = 0; a < N_ACT && action < 0; a++)
                if (mask[a]) action = a;
        if (action < 0) action = 0;
#ifdef BC_DUMP
        std::fprintf(stderr, "DUMP %d", prev_action);
        for (int i = 0; i < NCH * GC; i++) std::fprintf(stderr, " %d", (int)grid[i]);
        for (int a = 0; a < N_ACT; a++) std::fprintf(stderr, " %d", (int)mask[a]);
        for (int a = 0; a < N_ACT; a++) std::fprintf(stderr, " %.6g", ok ? logits[a] : 0.0f);
        std::fprintf(stderr, " %d\n", action);
        std::fprintf(stderr, "ACCPEAK %lld\n", (long long)ln::acc_peak);
#endif
        out += ok ? "INDICATOR lstm\n" : "INDICATOR NO NET\n";
        prev_action = action;
        emit(action, snap);
        flush_turn();
    }
}
