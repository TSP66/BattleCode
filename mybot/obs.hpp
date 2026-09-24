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

constexpr int N_MOVES = 3 + 9 + 27;
constexpr int N_SPLITS = 9;
constexpr int N_ACTIONS = N_MOVES + N_SPLITS;
constexpr std::array<int, N_SPLITS> SPLIT_K{2, 3, 4, 5, 6, 8, 12, 16, -1};  // -1 = half

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
    void scalars(unswbc::Controller const& ct, unswbc::Game const& game, float* out) const;
    void mask(unswbc::Controller const& ct, unswbc::Game const& game, std::uint8_t* out) const;

  private:
    std::vector<unswbc::Tile> const* tiles_ = nullptr;
    // our own segments, head first, as window cells; -1 past where the chain breaks
    std::array<int, CELLS> body_cell_{};
    int body_len_ = 0;

    void trace_body();
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

inline void Snapshot::scalars(unswbc::Controller const& ct, unswbc::Game const& game,
                              float* s) const {
    for (int i = 0; i < N_SCALARS; i++) s[i] = 0.0f;
    s[0] = (float)game.round_num / 500.0f;
    s[1] = (float)(length < 64 ? length : 64) / 64.0f;
    s[2] = (float)length;
    s[3] = (float)ct.get_unit_count() / (float)game.unit_limit;
    s[4 + facing] = 1.0f;
    s[8] = (float)hx / (float)w;
    s[9] = (float)hy / (float)h;
    s[10] = (float)w / 64.0f;
    s[11] = (float)h / 64.0f;
    auto const n = ct.sonar_messages.size();
    s[12] = (float)(n < MAX_MSGS ? n : MAX_MSGS);
    s[13] = ct.get_team().value == unswbc::Team::B ? 1.0f : 0.0f;
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

// Legality as far as the dragon can tell, a port of VecEnv::fill_mask. A
// portal ends the walk and counts as allowed, because the far side is not
// visible and guessing would teach the policy something false.
inline void Snapshot::mask(unswbc::Controller const& ct, unswbc::Game const& game,
                           std::uint8_t* m) const {
    for (int i = 0; i < N_ACTIONS; i++) m[i] = 0;

    std::array<int, 3> dirs{};
    for (int id = 0; id < N_MOVES; id++) {
        int x = hx, y = hy, len = length, dropped = 0, n_added = 0;
        std::array<int, 3> added{};
        bool legal = true;
        int const n = decode_move(id, facing, dirs);
        for (int s = 0; s < n; s++) {
            if (s > 0 && len <= 2) { legal = false; break; }
            int const here = cell_of(x, y);
            if (here < 0) { legal = false; break; }
            auto const kind = tile(here).edges[(std::size_t)dirs[(std::size_t)s]].edge_type;
            if (kind == unswbc::EdgeType::KELP) { legal = false; break; }
            if (kind == unswbc::EdgeType::PORTAL) break;   // cannot see where it lets out

            int const d = dirs[(std::size_t)s];
            int const nx = ((x + DX[(std::size_t)d]) % w + w) % w;
            int const ny = ((y + DY[(std::size_t)d]) % h + h) % h;
            int const there = cell_of(nx, ny);

            bool blocked = false;
            for (int j = 0; j < n_added; j++) if (added[(std::size_t)j] == there) blocked = true;
            if (!blocked && there >= 0) {
                auto const* part = tile(there).get_dragon();
                if (part != nullptr && part->dragon_id == my_id) {
                    blocked = true;
                    for (int j = 0; j < dropped; j++) {     // unless that tail has gone
                        int const seg = length - 1 - j;
                        if (seg >= 0 && seg < body_len_ && body_cell_[(std::size_t)seg] == there) {
                            blocked = false;
                            break;
                        }
                    }
                } else if (part != nullptr && !part->is_dragon_head) {
                    blocked = true;                          // a body is certain death
                }
            }
            if (blocked) { legal = false; break; }

            if (n_added < 3) added[(std::size_t)n_added++] = there;
            if (there >= 0 && tile(there).pearl) len++;
            else dropped++;
            if (s > 0) { dropped++; len--; }
            x = nx;
            y = ny;
        }
        m[id] = legal ? 1 : 0;
    }

    if (ct.get_unit_count() < game.unit_limit)
        for (int i = 0; i < N_SPLITS; i++) {
            int const k = SPLIT_K[(std::size_t)i] < 0 ? length / 2 : SPLIT_K[(std::size_t)i];
            m[N_MOVES + i] = (k >= 2 && k <= length - 2) ? 1 : 0;
        }
}

}  // namespace obs
