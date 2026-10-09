// Observation and action mask, built from one turn's protocol block.
//
// A port of bcsim/cpp/bc_obs.hpp, which is what the policy was trained
// against, by way of the NumPy version in tools/obs.py that was checked
// against the live engine. It has to agree with those exactly: a disagreement
// here means the deployed bot is answering a different question from the one
// it learned to answer.
//
// Everything is in the dragon's own frame -- the window is rotated so the
// dragon faces up, and the four directional channel groups are rolled to
// match -- so the policy never had to learn four copies of the same tactic.
#pragma once

#include "helper.hpp"

#include <array>
#include <cstdint>

namespace obs {

constexpr int VISION = 3;
constexpr int WINDOW = 7;
constexpr int CELLS = WINDOW * WINDOW;
constexpr int N_CHANNELS = 23;
constexpr int N_SCALARS = 14;
constexpr int MAX_MSGS = 4;
// The five protocol-3 echo counts (bc_core.hpp SonarEcho order: kelp, ally,
// ally_head, enemy, enemy_head). They sit at the very END of the scalar row,
// after the 694 remembered features, so [0, 708) stays byte for byte the row
// every earlier checkpoint was trained on. SONAR_DIRS is what the simulator
// divides them by, so a broadcast in all four directions maps onto [0, 1].
constexpr int N_ECHO = 5;
constexpr int SONAR_DIRS = 4;

// The action codec of the s2g15p simulator (bcsim/cpp/bc_vec.hpp CODEC_*): 39 relative paths of
// 1-3 steps, 9 splits, the self-kill (48), two splits sized from the end (49, 50: k = len - 2,
// len - 3) and 24 far sprints (51-74: one per window tile 4-6 steps away, walked by far_paths).
constexpr int N_MOVES = 3 + 9 + 27;
constexpr int N_SPLITS = 9;
constexpr int SUICIDE_ID = N_MOVES + N_SPLITS;            // 48
constexpr int XSPLIT_ID = SUICIDE_ID + 1;                 // 49
constexpr int N_XSPLITS = 2;
constexpr int FAR_ID = XSPLIT_ID + N_XSPLITS;             // 51
constexpr int N_FAR = 24;
constexpr int N_ACTIONS = FAR_ID + N_FAR;                 // 75
constexpr int FAR_MAX_STEPS = 8;
constexpr std::array<int, N_SPLITS> SPLIT_K{2, 3, 4, 5, 6, 8, 12, 16, -1};  // -1 = half
constexpr std::array<int, N_XSPLITS> XSPLIT_KEEP{2, 3};                      // k = len - keep

// The far targets as ego offsets (ox right, oy back; the dragon faces oy < 0), row-major from
// the front row: bc_vec.hpp FarTargets.
struct FarTargets {
    std::array<std::int8_t, N_FAR> ox{}, oy{};
    constexpr FarTargets() {
        int k = 0;
        for (int y = -3; y <= 3; y++)
            for (int x = -3; x <= 3; x++) {
                int const d = (x < 0 ? -x : x) + (y < 0 ? -y : y);
                if (d >= 4 && d <= 6) { ox[(std::size_t)k] = (std::int8_t)x; oy[(std::size_t)k] = (std::int8_t)y; k++; }
            }
    }
};
inline constexpr FarTargets FAR_TARGETS{};

// LocalChannel, in the order bc_vec.hpp declares it
constexpr int C_PEARL = 0, C_PEARL_TIME = 1, C_NEVER_SPAWN = 2;
constexpr int C_SELF_HEAD = 3, C_SELF_BODY = 4;
constexpr int C_ALLY_HEAD = 5, C_ALLY_BODY = 6;
constexpr int C_ENEMY_HEAD = 7, C_ENEMY_BODY = 8;
constexpr int C_FACE = 9, C_KELP = 13, C_PORTAL = 17;      // four channels each
constexpr int C_SELF_INDEX = 21, C_SELF_TAIL = 22;

constexpr std::array<int, 4> DX{0, 1, 0, -1};              // N, E, S, W
constexpr std::array<int, 4> DY{-1, 0, 1, 0};

inline int dir_index(unswbc::Direction d) {
    switch (d.value) {
        case unswbc::Direction::NORTH: return 0;
        case unswbc::Direction::EAST:  return 1;
        case unswbc::Direction::SOUTH: return 2;
        default:                       return 3;
    }
}

inline unswbc::Direction dir_of(int i) {
    static constexpr unswbc::Direction::Value v[4]{
        unswbc::Direction::NORTH, unswbc::Direction::EAST,
        unswbc::Direction::SOUTH, unswbc::Direction::WEST};
    return unswbc::Direction(v[i]);
}

// Output cell -> source cell in the protocol's window, one table per facing.
// This is ego_to_world from bc_obs.hpp, precomputed.
struct Tables {
    std::array<std::array<std::uint8_t, CELLS>, 4> perm{};
    // action id -> the turns it walks, low digit first: 0 straight, 1 left, 2 right
    std::array<std::array<std::int8_t, 3>, N_MOVES> turns{};
    std::array<std::int8_t, N_MOVES> n_steps{};

    constexpr Tables() {
        for (int facing = 0; facing < 4; facing++)
            for (int row = 0; row < WINDOW; row++)
                for (int col = 0; col < WINDOW; col++) {
                    int const ox = col - VISION, oy = row - VISION;
                    int wx = 0, wy = 0;
                    switch (facing) {
                        case 0: wx = ox;  wy = oy;  break;
                        case 1: wx = -oy; wy = ox;  break;
                        case 2: wx = -ox; wy = -oy; break;
                        default: wx = oy; wy = -ox; break;
                    }
                    perm[facing][row * WINDOW + col] =
                        (std::uint8_t)((wy + VISION) * WINDOW + (wx + VISION));
                }
        for (int id = 0; id < N_MOVES; id++) {
            int n = 0, rest = 0;
            if (id < 3) { n = 1; rest = id; }
            else if (id < 12) { n = 2; rest = id - 3; }
            else { n = 3; rest = id - 12; }
            n_steps[id] = (std::int8_t)n;
            for (int i = 0; i < n; i++) { turns[id][i] = (std::int8_t)(rest % 3); rest /= 3; }
        }
    }
};

inline constexpr Tables TABLES{};

// The world directions a move action walks, given where the dragon points now.
inline int decode_move(int id, int facing, std::array<int, 3>& out) {
    int f = facing;
    int const n = TABLES.n_steps[id];
    for (int i = 0; i < n; i++) {
        int const t = TABLES.turns[id][i];
        if (t == 1) f = (f + 3) % 4;
        else if (t == 2) f = (f + 1) % 4;
        out[i] = f;
    }
    return n;
}

// One turn's block, indexed the way the observation and the mask need it.
class Snapshot {
  public:
    int w = 0, h = 0, hx = 0, hy = 0, facing = 0, length = 0, my_id = 0;
    char my_team = 'A';

    void build(unswbc::Controller const& ct, unswbc::Game const& game) {
        w = game.width;
        h = game.height;
        auto const pos = ct.get_position();
        hx = pos.x;
        hy = pos.y;
        facing = dir_index(ct.get_dir());
        length = ct.get_length();
        my_id = ct.get_id();
        my_team = (char)ct.get_team().value;
        tiles_ = &ct.get_tiles();
        trace_body();
    }

    // The protocol's window index of a board position, or -1 when outside it.
    int cell_of(int x, int y) const {
        int const dx = ((x - hx + VISION) % w + w) % w;
        int const dy = ((y - hy + VISION) % h + h) % h;
        if (dx >= WINDOW || dy >= WINDOW) return -1;
        return dy * WINDOW + dx;
    }

    unswbc::Tile const& tile(int cell) const { return (*tiles_)[(std::size_t)cell]; }

    void local(float* out) const;
    // our own segments as far as the window shows them, head first: window cells
    int chain_len() const { return body_len_; }
    // The whole body as the bot tracks it (main.cpp track_body), as window cells (-1
    // outside the window), so mask() can see a tail leaving a cell even when the
    // visible chain does not reach it -- which is what the simulator's mask knows.
    void set_body(int const* cells, int n) {
        full_n_ = n < 128 ? n : 128;
        for (int i = 0; i < full_n_; i++) full_[(std::size_t)i] = cells[i];
    }
    int chain_cell(int i) const { return body_cell_[(std::size_t)i]; }
    // The legal-move mask over N_ACTIONS (bc_obs.hpp fill_mask), with the queen guard when
    // `queen_guard` (the dragon is a queen and the policy trained with BC_QUEEN_GUARD=1).
    // Also fills far_n / far_dirs: each far target's path (world directions), 0 = none.
    void mask(unswbc::Controller const& ct, unswbc::Game const& game, bool queen_guard, std::uint8_t* out);
    // The queen dead-end mask (bc_obs.hpp VecEnv::queen_deadend_mask, BC_QUEEN_DEADEND=level): after mask()
    void queen_deadend(std::uint8_t* m, int level = 1) const;
    std::array<int, N_FAR> far_n{};
    std::array<int, N_FAR * FAR_MAX_STEPS> far_dirs{};

  private:
    std::vector<unswbc::Tile> const* tiles_ = nullptr;
    // our own segments, head first, as window cells; -1 past where the chain breaks
    std::array<int, CELLS> body_cell_{};
    int body_len_ = 0;
    std::array<int, 128> full_{};
    int full_n_ = -1;

    void trace_body();
    // L: her length as if she had just split (-1 = as she is); her segments from index L on are the
    // child's, another dragon's (its head at index length - 1)
    bool path_legal(int const* dirs, int n, bool heads_block, int L = -1) const;
    int own_index(int c) const;
    int move_fatal(int const* dirs, int n, int L, bool own_walls) const;
    void far_paths();
};

// Walks our own body outward from the head.
//
// The protocol never numbers our segments, but every body segment points at
// the segment ahead of it (bc_core.hpp sets seg_dir that way), so the chain
// can be followed backwards from the head. A segment pointing through a
// portal lands somewhere we cannot see, which ends the walk; the segments
// past it stay unnumbered, which costs two channels on tiles we can see
// anyway.
inline void Snapshot::trace_body() {
    // back[cell] = the cell of the segment that points at it
    std::array<std::int8_t, CELLS> back{};
    back.fill(-1);
    for (int i = 0; i < CELLS; i++) {
        auto const& t = tile(i);
        auto const* part = t.get_dragon();
        if (part == nullptr || part->dragon_id != my_id || part->is_dragon_head) continue;
        int const d = dir_index(part->dir);
        if (t.edges[(std::size_t)d].edge_type != unswbc::EdgeType::EMPTY) continue;
        int const nx = ((t.position.x + DX[(std::size_t)d]) % w + w) % w;
        int const ny = ((t.position.y + DY[(std::size_t)d]) % h + h) % h;
        int const cell = cell_of(nx, ny);
        if (cell >= 0) back[(std::size_t)cell] = (std::int8_t)i;
    }

    body_cell_.fill(-1);
    full_n_ = -1;
    std::array<bool, CELLS> seen{};
    int cur = cell_of(hx, hy);
    body_len_ = 0;
    while (cur >= 0 && !seen[(std::size_t)cur] && body_len_ < CELLS) {
        seen[(std::size_t)cur] = true;
        body_cell_[(std::size_t)body_len_++] = cur;
        cur = back[(std::size_t)cur];
    }
}

inline void Snapshot::local(float* out) const {
    float world[N_CHANNELS][CELLS]{};

    for (int i = 0; i < CELLS; i++) {
        auto const& t = tile(i);
        if (t.pearl) world[C_PEARL][i] = 1.0f;
        if (t.pearl_time < 0) world[C_NEVER_SPAWN][i] = 1.0f;
        else world[C_PEARL_TIME][i] = (float)(t.pearl_time < 99 ? t.pearl_time : 99) / 99.0f;

        if (auto const* part = t.get_dragon()) {
            bool const head = part->is_dragon_head;
            if (part->dragon_id == my_id)
                world[head ? C_SELF_HEAD : C_SELF_BODY][i] = 1.0f;
            else if ((char)part->team.value == my_team)
                world[head ? C_ALLY_HEAD : C_ALLY_BODY][i] = 1.0f;
            else
                world[head ? C_ENEMY_HEAD : C_ENEMY_BODY][i] = 1.0f;
            world[C_FACE + dir_index(part->dir)][i] = 1.0f;
        }

        for (int d = 0; d < 4; d++) {
            auto const kind = t.edges[(std::size_t)d].edge_type;
            if (kind == unswbc::EdgeType::KELP) world[C_KELP + d][i] = 1.0f;
            else if (kind == unswbc::EdgeType::PORTAL) world[C_PORTAL + d][i] = 1.0f;
        }
    }

    float const denom = (float)(length - 1 > 1 ? length - 1 : 1);
    for (int idx = 0; idx < body_len_; idx++) {
        int const cell = body_cell_[(std::size_t)idx];
        world[C_SELF_INDEX][cell] = (float)idx / denom;
        if (idx == length - 1) world[C_SELF_TAIL][cell] = 1.0f;
    }

    // rotate into the dragon's frame: cells through the permutation, and the
    // three four-channel groups rolled so N,E,S,W become fwd,right,back,left
    auto const& p = TABLES.perm[(std::size_t)facing];
    for (int c = 0; c < N_CHANNELS; c++) {
        int src = c;
        if (c >= C_FACE && c < C_FACE + 4) src = C_FACE + (c - C_FACE + facing) % 4;
        else if (c >= C_KELP && c < C_KELP + 4) src = C_KELP + (c - C_KELP + facing) % 4;
        else if (c >= C_PORTAL && c < C_PORTAL + 4) src = C_PORTAL + (c - C_PORTAL + facing) % 4;
        float* dst = out + (std::size_t)c * CELLS;
        float const* row = world[src];
        for (int i = 0; i < CELLS; i++) dst[i] = row[p[(std::size_t)i]];
    }
}

// The five echo counts, normalised exactly as bc_obs.hpp does:
//     sc[SC_ECHO_AT + k] = echo[k] / SONAR_DIRS
// Written at the end of the row, so `out` is scalars + N_SCALARS + N_EXTRA.
// Absent on a dragon's very first block, because it has not declared protocol 3
// yet; the helper leaves the counts zero then, which is what the simulator has
// at a dragon's first turn too.
inline void echo_scalars(unswbc::Controller const& ct, float* out) {
    unswbc::SonarEchoes const e = ct.get_sonar_echoes();
    int const counts[N_ECHO] = {e.kelp, e.ally, e.ally_head, e.enemy, e.enemy_head};
    for (int k = 0; k < N_ECHO; k++) out[k] = (float)counts[k] / (float)SONAR_DIRS;
}

// Legality as far as the dragon can tell, a port of VecEnv::path_legal: kelp and visible bodies
// block, a portal hides what is beyond it so the step counts as allowed, and steps past the free
// ones have to be paid for. heads_block (the queen guard): another dragon's head blocks too, and
// past a portal the paid steps must still be affordable, counting no pearls over there.
inline bool Snapshot::path_legal(int const* dirs, int n, bool heads_block, int L) const {
    bool const as_split = L >= 0 && L != length;
    if (L < 0) L = length;
    int x = hx, y = hy, len = L, dropped = 0, n_added = 0;
    int const free = (L + 3) / 4;   // steps free of a segment (unswbc 1.2.3)
    std::array<int, FAR_MAX_STEPS> added{};
    for (int s = 0; s < n; s++) {
        if (s >= free && len <= 2) return false;
        int const here = cell_of(x, y);
        if (here < 0) return false;
        int const d = dirs[s];
        auto const kind = tile(here).edges[(std::size_t)d].edge_type;
        if (kind == unswbc::EdgeType::KELP) return false;
        if (kind == unswbc::EdgeType::PORTAL) {        // cannot see where it lets out ...
            if (heads_block)                           // ... but a guarded queen's paid steps must stay affordable
                for (int r = s; r < n; r++) {
                    if (r >= free && len <= 2) return false;
                    if (r >= free) len--;
                }
            return true;
        }
        int const nx = ((x + DX[(std::size_t)d]) % w + w) % w;
        int const ny = ((y + DY[(std::size_t)d]) % h + h) % h;
        int const there = cell_of(nx, ny);

        bool blocked = false;
        for (int j = 0; j < n_added; j++) if (added[(std::size_t)j] == there) blocked = true;
        if (!blocked && there >= 0) {
            auto const* part = tile(there).get_dragon();
            int const j_own = as_split && part != nullptr && part->dragon_id == my_id ? own_index(there) : -1;
            if (j_own >= L) {                            // the split-off child's
                if (heads_block || j_own != length - 1) blocked = true;
            } else if (part != nullptr && part->dragon_id == my_id) {
                blocked = true;
                for (int j = 0; j < dropped; j++) {     // unless that tail has gone
                    int const seg = L - 1 - j;
                    bool const at_seg = full_n_ >= 0
                        ? (seg >= 0 && seg < full_n_ && full_[(std::size_t)seg] == there)
                        : (seg >= 0 && seg < body_len_ && body_cell_[(std::size_t)seg] == there);
                    if (at_seg) {
                        blocked = false;
                        break;
                    }
                }
            } else if (part != nullptr && (heads_block || !part->is_dragon_head)) {
                blocked = true;                          // another dragon's body is certain death
            }
        }
        if (blocked) return false;

        if (n_added < FAR_MAX_STEPS) added[(std::size_t)n_added++] = there;
        if (there >= 0 && tile(there).pearl) len++;
        else dropped++;
        if (s >= free) { dropped++; len--; }
        x = nx;
        y = ny;
    }
    return true;
}

// The far sprints' paths, a port of VecEnv::far_paths: ONE breadth-first search over the 7x7
// window in the dragon's own frame, from its head, stepping forward, left, right, back in that
// order. A step must cross an open edge into a free tile; another dragon's head may end a path
// but is never passed through; bodies (tails included) block as they stand.
inline void Snapshot::far_paths() {
    static constexpr int EGO_DIR[4] = {0, 3, 1, 2};       // forward (ego N), left (W), right (E), back (S)
    static constexpr int EDX[4] = {0, 1, 0, -1}, EDY[4] = {-1, 0, 1, 0};
    std::array<std::int8_t, CELLS> from{}, how{}, depth{};
    from.fill(-1);
    auto ego = [&](int ox, int oy) { return (oy + VISION) * WINDOW + (ox + VISION); };
    // ego offset -> the protocol's window cell (world offset from the head)
    auto win = [&](int ox, int oy) {
        int wx = 0, wy = 0;
        switch (facing) {
            case 0: wx = ox;  wy = oy;  break;
            case 1: wx = -oy; wy = ox;  break;
            case 2: wx = -ox; wy = -oy; break;
            default: wx = oy; wy = -ox; break;
        }
        return (wy + VISION) * WINDOW + (wx + VISION);
    };
    std::array<int, CELLS> queue{};
    int qh = 0, qt = 0;
    int const start = ego(0, 0);
    from[(std::size_t)start] = (std::int8_t)start;
    depth[(std::size_t)start] = 0;
    queue[(std::size_t)qt++] = start;
    while (qh < qt) {
        int const c = queue[(std::size_t)qh++];
        int const ox = c % WINDOW - VISION, oy = c / WINDOW - VISION;
        if (depth[(std::size_t)c] >= FAR_MAX_STEPS) continue;
        auto const& here = tile(win(ox, oy));
        for (int j = 0; j < 4; j++) {
            int const ed = EGO_DIR[j];
            int const nox = ox + EDX[ed], noy = oy + EDY[ed];
            if (nox < -VISION || nox > VISION || noy < -VISION || noy > VISION) continue;
            int const nc = ego(nox, noy);
            if (from[(std::size_t)nc] >= 0) continue;
            int const wd = (ed + facing) & 3;
            if (here.edges[(std::size_t)wd].edge_type != unswbc::EdgeType::EMPTY) continue;
            auto const* part = tile(win(nox, noy)).get_dragon();
            if (part != nullptr && (part->dragon_id == my_id || !part->is_dragon_head)) continue;   // a body
            from[(std::size_t)nc] = (std::int8_t)c;
            how[(std::size_t)nc] = (std::int8_t)wd;
            depth[(std::size_t)nc] = (std::int8_t)(depth[(std::size_t)c] + 1);
            if (part == nullptr) queue[(std::size_t)qt++] = nc;      // a head ends a path, never passed
        }
    }
    for (int k = 0; k < N_FAR; k++) {
        int const goal = ego(FAR_TARGETS.ox[(std::size_t)k], FAR_TARGETS.oy[(std::size_t)k]);
        far_n[(std::size_t)k] = 0;
        if (from[(std::size_t)goal] < 0) continue;
        int const n = depth[(std::size_t)goal];
        int* dirs = far_dirs.data() + (std::size_t)k * FAR_MAX_STEPS;
        int i = n;
        for (int c = goal; c != start; c = from[(std::size_t)c]) dirs[--i] = how[(std::size_t)c];
        far_n[(std::size_t)k] = n;
    }
}

// The legal-move mask, a port of VecEnv::fill_mask.
inline void Snapshot::mask(unswbc::Controller const& ct, unswbc::Game const& game, bool queen_guard,
                           std::uint8_t* m) {
    for (int i = 0; i < N_ACTIONS; i++) m[i] = 0;
    m[SUICIDE_ID] = 1;                                  // always available

    std::array<int, 3> dirs3{};
    int dirs[FAR_MAX_STEPS];
    for (int id = 0; id < N_MOVES; id++) {
        int const n = decode_move(id, facing, dirs3);
        for (int s = 0; s < n; s++) dirs[s] = dirs3[(std::size_t)s];
        m[id] = path_legal(dirs, n, false) ? 1 : 0;
    }
    far_paths();
    for (int k = 0; k < N_FAR; k++)
        m[FAR_ID + k] = (far_n[(std::size_t)k] > 0 &&
                         path_legal(far_dirs.data() + (std::size_t)k * FAR_MAX_STEPS, far_n[(std::size_t)k], false)) ? 1 : 0;

    bool const room = ct.get_unit_count() < game.unit_limit;
    for (int i = 0; i < N_SPLITS; i++) {
        int const k = SPLIT_K[(std::size_t)i] < 0 ? length / 2 : SPLIT_K[(std::size_t)i];
        m[N_MOVES + i] = (room && k >= 2 && k <= length - 2) ? 1 : 0;
    }
    for (int i = 0; i < N_XSPLITS; i++) {
        int const k = length - XSPLIT_KEEP[(std::size_t)i];
        m[XSPLIT_ID + i] = (room && k >= 2 && k <= length - 2) ? 1 : 0;
    }

    // the queen guard (bc_vec.hpp VecConfig::queen_guard): no self-kill, no move onto a head --
    // unless that leaves her nothing, when she keeps the ordinary mask
    if (queen_guard) {
        std::uint8_t guarded[N_ACTIONS];
        for (int i = 0; i < N_ACTIONS; i++) guarded[i] = m[i];
        guarded[SUICIDE_ID] = 0;
        for (int id = 0; id < N_MOVES; id++) {
            if (!guarded[id]) continue;
            int const n = decode_move(id, facing, dirs3);
            for (int s = 0; s < n; s++) dirs[s] = dirs3[(std::size_t)s];
            if (!path_legal(dirs, n, true)) guarded[id] = 0;
        }
        for (int k = 0; k < N_FAR; k++)
            if (guarded[FAR_ID + k] &&
                !path_legal(far_dirs.data() + (std::size_t)k * FAR_MAX_STEPS, far_n[(std::size_t)k], true))
                guarded[FAR_ID + k] = 0;
        bool any = false;
        for (int i = 0; i < N_ACTIONS; i++) any = any || guarded[i];
        if (any)
            for (int i = 0; i < N_ACTIONS; i++) m[i] = guarded[i];
    }
}

// Our segment index of window cell c from the tracked body (set_body; the visible chain without it),
// -1 when unknown (a guessed or unseen cell).
inline int Snapshot::own_index(int c) const {
    if (c < 0) return -1;
    if (full_n_ >= 0) {
        for (int i = 0; i < full_n_; i++) if (full_[(std::size_t)i] == c) return i;
        return -1;
    }
    for (int i = 0; i < body_len_; i++) if (body_cell_[(std::size_t)i] == c) return i;
    return -1;
}

// One move judged as VecEnv::queen_move_fatal judges it, from what the window shows: walked as
// path_legal walks it (the cells it adds, how many tail cells it vacates, pearls, paid steps), then
// queen_in_dead_end's walk from the new head: one way at every tile until none, all within VISION of
// the head she decides from, no portal, no dragon beside it. An open edge out of the window is never a
// dead end (the simulator's walk leaves the window there too). A path through a portal is not judged
// (0). L: her length (after a split on a copy: the rest is the child's). own_walls (level 2): her own
// segment beside the tube at walk depth k is a wall while still hers on her k+1-th move from there,
// len - s > k + 1; a segment of unknown index stays "might move".
inline int Snapshot::move_fatal(int const* dirs, int n, int L, bool own_walls) const {
    int const free_steps = (L + 3) / 4;
    int x = hx, y = hy, dropped = 0, n_added = 0;
    std::array<int, FAR_MAX_STEPS> added{};
    bool portal = false, lost = false;
    for (int s = 0; s < n && !portal && !lost; s++) {
        int const here = cell_of(x, y);
        if (here < 0) { lost = true; break; }
        if (tile(here).edges[(std::size_t)dirs[s]].edge_type == unswbc::EdgeType::PORTAL) { portal = true; break; }
        x = ((x + DX[(std::size_t)dirs[s]]) % w + w) % w;
        y = ((y + DY[(std::size_t)dirs[s]]) % h + h) % h;
        int const there = cell_of(x, y);
        if (n_added < FAR_MAX_STEPS) added[(std::size_t)n_added++] = there;
        if (there >= 0 && tile(there).pearl) {} else dropped++;
        if (s >= free_steps) dropped++;
    }
    if (portal || lost) return 0;
    int const L2 = L + n_added - dropped;                  // her length after the move
    // -1 free, -2 someone who might move, else her own segment index after the move
    auto occupant = [&](int c) {
        for (int j = 0; j < n_added; j++) if (added[(std::size_t)j] == c) return n_added - 1 - j;
        auto const* part = tile(c).get_dragon();
        if (part == nullptr) return -1;
        if (part->dragon_id != my_id) return -2;
        if (!own_walls && L == length) {                   // level 1, exactly as it was
            for (int j = 0; j < dropped; j++) {            // her tail cells the move vacated
                int const seg = length - 1 - j;
                bool const at_seg = full_n_ >= 0
                    ? (seg >= 0 && seg < full_n_ && full_[(std::size_t)seg] == c)
                    : (seg >= 0 && seg < body_len_ && body_cell_[(std::size_t)seg] == c);
                if (at_seg) return -1;
            }
            return -2;
        }
        int const j = own_index(c);
        if (j < 0 || j >= L) return -2;                    // unknown, or the child's
        if (j >= L - dropped) return -1;                   // a tail cell the move vacated
        return own_walls ? j + n_added : -2;
    };
    int came = dirs[n - 1];
    for (int depth = 0; depth <= 2 * WINDOW; depth++) {
        int ox = x - hx, oy = y - hy;
        if (ox > w / 2) ox -= w;
        if (ox < -(w - 1) / 2) ox += w;
        if (oy > h / 2) oy -= h;
        if (oy < -(h - 1) / 2) oy += h;
        if (ox < -VISION || ox > VISION || oy < -VISION || oy > VISION) return 0;
        int const c = cell_of(x, y);
        int const back = (came + 2) & 3;
        int ways = 0, nx = 0, ny = 0, nd = 0;
        for (int dir = 0; dir < 4; dir++) {
            if (dir == back) continue;
            auto const kind = tile(c).edges[(std::size_t)dir].edge_type;
            if (kind == unswbc::EdgeType::KELP) continue;
            if (kind == unswbc::EdgeType::PORTAL) return 0;
            int const tx = ((x + DX[(std::size_t)dir]) % w + w) % w, ty = ((y + DY[(std::size_t)dir]) % h + h) % h;
            int const tc = cell_of(tx, ty);
            if (tc < 0) return 0;
            int const occ = occupant(tc);
            if (occ == -2) return 0;
            if (occ >= 0) {
                if (own_walls && L2 - occ > depth + 1) continue;   // still her body when she would get there
                return 0;
            }
            ways++;
            nx = tx; ny = ty; nd = dir;
        }
        if (ways > 1) return 0;
        if (ways == 0) return 1;
        x = nx; y = ny; came = nd;
    }
    return 0;
}

// VecEnv::queen_deadend_mask, from what the window shows (move_fatal). Fatal moves go only while
// another move is not fatal. Level 1: splits and the self-kill are left alone. Level 2: her own body
// walls the dead end, and when every move is fatal each legal split is tried: it rescues her if, as
// the shorter queen (the rest the child's), some move path_legal allows (heads blocking) is not
// fatal; if one does, only the rescuing splits stay.
inline void Snapshot::queen_deadend(std::uint8_t* m, int level) const {
    std::array<std::uint8_t, N_ACTIONS> fatal{};
    int n_fatal = 0, n_ok = 0;
    bool const own_walls = level >= 2;
    std::array<int, 3> d3{};
    int dirs[FAR_MAX_STEPS];
    auto path_of = [&](int id) {
        int n = 0;
        if (id < N_MOVES) {
            n = decode_move(id, facing, d3);
            for (int s = 0; s < n; s++) dirs[s] = d3[(std::size_t)s];
        } else if (id >= FAR_ID && id < FAR_ID + N_FAR) {
            n = far_n[(std::size_t)(id - FAR_ID)];
            for (int s = 0; s < n; s++) dirs[s] = far_dirs[(std::size_t)((id - FAR_ID) * FAR_MAX_STEPS + s)];
        }
        return n;
    };
    for (int id = 0; id < N_ACTIONS; id++) {
        if (!m[id]) continue;
        int const n = path_of(id);
        if (n == 0) continue;                              // a split or the self-kill
        if (move_fatal(dirs, n, length, own_walls)) { fatal[(std::size_t)id] = 1; n_fatal++; }
        else n_ok++;
    }
    if (n_fatal > 0 && n_ok > 0) {
        for (int id = 0; id < N_ACTIONS; id++)
            if (fatal[(std::size_t)id]) m[id] = 0;
        return;
    }
    if (level < 2 || n_fatal == 0) return;
    // every move is fatal: which splits leave her a move that is not?
    std::array<std::uint8_t, N_ACTIONS> rescue{};
    int n_rescue = 0;
    for (int id = 0; id < N_ACTIONS; id++) {
        if (!m[id]) continue;
        int k = 0;
        if (id >= N_MOVES && id < N_MOVES + N_SPLITS)
            k = SPLIT_K[(std::size_t)(id - N_MOVES)] < 0 ? length / 2 : SPLIT_K[(std::size_t)(id - N_MOVES)];
        else if (id >= XSPLIT_ID && id < XSPLIT_ID + N_XSPLITS)
            k = length - XSPLIT_KEEP[(std::size_t)(id - XSPLIT_ID)];
        else continue;
        int const L = length - k;
        bool out = false;
        for (int mid = 0; mid < N_ACTIONS && !out; mid++) {
            int const n = path_of(mid);
            if (n == 0) continue;
            if (path_legal(dirs, n, true, L) && !move_fatal(dirs, n, L, true)) out = true;
        }
        if (out) { rescue[(std::size_t)id] = 1; n_rescue++; }
    }
    if (n_rescue > 0)
        for (int id = 0; id < N_ACTIONS; id++) m[id] = m[id] && rescue[(std::size_t)id];
}

}  // namespace obs
