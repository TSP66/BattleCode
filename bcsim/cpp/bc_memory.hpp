// What a dragon remembers of the map, as extra network inputs.
//
// The 7x7 window and 14 scalars cannot express a team that plays off a
// remembered map, so train/clone_features.py adds two inputs built from a
// dragon's own earlier turns: `mem` (676, the 13x13 around the head in its own
// frame) and `memfar` (18, the whole remembered map summarised). They are worth
// +2.9 points of held-out accuracy (DISTILL_DEVTEST.md), and mybot/memory.hpp
// already computes them for the deployed bot.
//
// This is the same thing inside the simulator, so PPO can train with memory
// instead of paying for a Python tracker in the rollout loop (measured: 22.8k
// agent-turns/s against the ~43k rollout needs, and 26 MB per env).
//
// THE AUTHORITY IS train/clone_features.py's MemoryTracker, not mybot. The
// clones' weights were fitted through the Python tracker, so where the two
// could differ, the Python one is right. In particular it maps a window cell to
// the world by ego offset from the head and does NOT model portal visibility;
// this copies that, quirk included.
//
// Two representation details exist because the trainer stored the features
// quantised, and the network was fitted to the quantised values:
//   * the decayed ages are a byte (uint8, /255), so they come from a table
//     built with the same rounding rather than from expf;
//   * memfar was stored as float16, so each of its 18 values is rounded to
//     half precision.
//
// Memory is 6 bytes a cell and sized to the map, because a trained policy
// saturates the 64-unit limit on 64x64 maps: at 1024 envs the naive int32
// layout is 7.5 GB, this is under 1 GB for a normal map mix.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace bc {

namespace mem_cfg {
constexpr int R = 6;                          // mem is the 13x13 around the head
constexpr int SIDE = 2 * R + 1;
constexpr int CELLS = SIDE * SIDE;            // 169
constexpr int N_MEM = 4 * CELLS;              // 676
constexpr int K = 3;                          // nearest expected pearls memfar names
constexpr int N_FAR = 4 * K + 6;              // 18
constexpr int N_EXTRA = N_MEM + N_FAR;        // 694
constexpr int NEVER = -10000;                 // "not seen", as the builders write it
constexpr int UNKNOWN_AGE = 10000;            // MemoryTracker: known = age < 10_000
// 255 * exp(-age/32) and 255 * exp(-age/16) round to zero past these
constexpr int AGE_MAX = 200;
constexpr int VISIT_MAX = 100;
}  // namespace mem_cfg

// The same remembered map as `mem`, given back as planes instead of a flat bag
// of numbers, and reaching further.
//
// `mem` crops radius 6 and hands 676 floats to a Linear(708, 128) that is 0.55%
// of the forward pass: the geometry is thrown away and 92% of the compute goes
// on the inner 49 cells. These are the same arrays, read as two stacked scales
// in the head's own frame:
//   * near, stride 1, radius 7   -- the 15x15 immediately around the head;
//   * far,  stride POOL, mean-pooled, reaching radius 4 * 7 + 1 = 29.
// Six channels each, `mem`'s four plus the respawn timer and where an enemy was
// last seen -- the hole that mattered, since `mem` remembers pearls and kelp
// but forgets that it ever saw a dragon.
namespace wide_cfg {
constexpr int R = 7;                          // 15x15 at either scale
constexpr int SIDE = 2 * R + 1;
constexpr int CELLS = SIDE * SIDE;            // 225
constexpr int CH = 6;                         // per scale
constexpr int N_WIDE = 2 * CH * CELLS;        // 2700
constexpr int POOL = 4;                       // world cells a far cell averages, per axis
}  // namespace wide_cfg

// The LSTM policy's input (bcsim/train/lstm_net.py CHANNELS): a 14x14 grid in
// the dragon's frame, head at row 7 col 7, rendered each turn from the memory
// below. Static terrain and the decayed memory channels come from here; the
// live 7x7, the self channels and the global planes are filled by bc_obs.hpp.
// Decays are Q12-rounded (x4096) so the deployed int16 bot reproduces them.
namespace grid_cfg {
constexpr int G = 14, HALF = 7, CELLS = G * G;
constexpr int CH = 38;
constexpr int N = CH * CELLS;
constexpr int AGE_CAP = 256;                   // older than this reads as 0
enum : int {
    KELP = 0, PORTAL = 4, NEVER_SPAWNS = 8,
    PEARL = 9, PEARL_TIMER = 10, ALLY_HEAD = 11, ALLY_BODY = 12, ENEMY_HEAD = 13,
    ENEMY_BODY = 14, SEG_DIR = 15,
    SELF_BODY = 19, SELF_INDEX = 20, SELF_TAIL = 21,
    SEEN = 22, PEARL_EXPECTED = 23, PEARL_TIMER_PROJ = 24, ENEMY_MEM = 25, ALLY_MEM = 26,
    VISITED = 27,
    ROUND = 28, LENGTH = 29, UNITS = 30, MAP_W = 31, MAP_H = 32, ECHO = 33,
};
}  // namespace grid_cfg

struct GridDecay {
    float seen[grid_cfg::AGE_CAP]{}, pearl[grid_cfg::AGE_CAP]{},
          dragon[grid_cfg::AGE_CAP]{}, visit[grid_cfg::AGE_CAP]{};
    GridDecay() {
        auto q = [](double tau, int a) { return (float)(std::nearbyint(4096.0 * std::exp(-a / tau)) / 4096.0); };
        for (int a = 0; a < grid_cfg::AGE_CAP; a++) {
            seen[a] = q(32.0, a);
            pearl[a] = q(64.0, a);
            dragon[a] = q(8.0, a);
            visit[a] = q(16.0, a);
        }
    }
    static float at(float const* t, int age) {
        return (unsigned)age < (unsigned)grid_cfg::AGE_CAP ? t[age] : 0.0f;
    }
};

inline const GridDecay GRID_DECAY{};

inline int mem_wrap(int v, int m) { return (v % m + m) % m; }

// ego (row, col) of the 13x13 -> world offset, per facing. The same rotation
// bc_obs.hpp's ego_to_world applies to the window, at radius 6 instead of 3.
struct MemEgoTable {
    std::array<std::array<std::int8_t, mem_cfg::CELLS>, 4> dx{}, dy{};
    constexpr MemEgoTable() {
        for (int f = 0; f < 4; f++)
            for (int row = 0; row < mem_cfg::SIDE; row++)
                for (int col = 0; col < mem_cfg::SIDE; col++) {
                    int const ox = col - mem_cfg::R, oy = row - mem_cfg::R;
                    int wx = 0, wy = 0;
                    switch (f) {
                        case 0: wx = ox;  wy = oy;  break;
                        case 1: wx = -oy; wy = ox;  break;
                        case 2: wx = -ox; wy = -oy; break;
                        default: wx = oy; wy = -ox; break;
                    }
                    dx[f][(std::size_t)(row * mem_cfg::SIDE + col)] = (std::int8_t)wx;
                    dy[f][(std::size_t)(row * mem_cfg::SIDE + col)] = (std::int8_t)wy;
                }
    }
};

inline constexpr MemEgoTable MEM_EGO{};

// The same rotation MemEgoTable bakes in, for offsets past radius 6 and for the
// pooled scale, whose sub-cell offsets are too many to table usefully.
inline void mem_ego_to_world(int facing, int ox, int oy, int& wx, int& wy) {
    switch (facing) {
        case 0: wx = ox;  wy = oy;  break;
        case 1: wx = -oy; wy = ox;  break;
        case 2: wx = -ox; wy = -oy; break;
        default: wx = oy; wy = -ox; break;
    }
}

// The decay tables, built once with the trainer's rounding.
struct MemDecay {
    float seen[mem_cfg::AGE_MAX]{};
    float visit[mem_cfg::VISIT_MAX]{};
    MemDecay() {
        for (int a = 0; a < mem_cfg::AGE_MAX; a++)
            seen[a] = (float)(std::nearbyint(255.0 * std::exp(-(double)a / 32.0)) / 255.0);
        for (int a = 0; a < mem_cfg::VISIT_MAX; a++)
            visit[a] = (float)(std::nearbyint(255.0 * std::exp(-(double)a / 16.0)) / 255.0);
    }
};

inline const MemDecay MEM_DECAY{};

// Rounds to what float16 holds, which is how memfar was stored for training.
// Round to nearest, ties to even, on the top 10 mantissa bits. Every memfar
// value is 0 or between 2^-12 and 4, so this never needs a subnormal.
inline float mem_half(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    std::uint32_t const lsb = 1u << 13;
    std::uint32_t const rest = u & (lsb - 1);
    u -= rest;
    if (rest > lsb / 2 || (rest == lsb / 2 && (u & lsb) != 0)) u += lsb;
    float y = 0.0f;
    std::memcpy(&y, &u, sizeof y);
    return y;
}

// One dragon's memory. Cleared when the slot is handed to a new dragon, so a
// split child starts empty -- which is what the trainer assumed and what the
// deployed bot does, since the judge runs one process per dragon.
struct DragonMemory {
    int w = 0, h = 0;
    std::vector<std::int16_t> seen, visit;   // round last seen / stood on, or NEVER
    std::vector<std::int8_t> cd;             // -1 never spawns, else the timer capped at 99
    std::vector<std::uint8_t> flags;         // bit 0 pearl, bit 1 kelp
    // Round an enemy segment was last standing here, or NEVER. Only `wide`
    // reads it: `mem` and `memfar` are the trainer's features and must not
    // change, so this costs the old inputs nothing.
    std::vector<std::int16_t> foe;
    // For the 14x14 grid only, so mem, memfar and wide stay bit-identical:
    // where an ally segment was last seen, and which sides of the cell carry
    // an edge, in WORLD directions (bits 0-3 kelp N E S W, bits 4-7 portal).
    std::vector<std::int16_t> ally;
    std::vector<std::uint8_t> sides;

    static constexpr std::uint8_t PEARL = 1, KELP = 2;

    void reset(int w_, int h_) {
        w = w_;
        h = h_;
        std::size_t const n = (std::size_t)w * (std::size_t)h;
        seen.assign(n, (std::int16_t)mem_cfg::NEVER);
        visit.assign(n, (std::int16_t)mem_cfg::NEVER);
        cd.assign(n, (std::int8_t)-1);
        flags.assign(n, 0);
        foe.assign(n, (std::int16_t)mem_cfg::NEVER);
        ally.assign(n, (std::int16_t)mem_cfg::NEVER);
        sides.assign(n, 0);
    }

    std::size_t at(int x, int y) const {
        return (std::size_t)mem_wrap(y, h) * (std::size_t)w + (std::size_t)mem_wrap(x, w);
    }

    // One window cell, as the observation would have shown it. `timer` is the
    // raw respawn counter: negative means the tile never spawns.
    void see(int x, int y, int round, bool pearl, int timer, bool kelp) {
        std::size_t const k = at(x, y);
        seen[k] = (std::int16_t)round;
        cd[k] = timer < 0 ? (std::int8_t)-1 : (std::int8_t)std::min(timer, 99);
        std::uint8_t f = (std::uint8_t)(flags[k] & KELP);   // kelp is never cleared
        if (pearl) f |= PEARL;
        if (kelp) f |= KELP;
        flags[k] = f;
    }

    void stand(int x, int y, int round) { visit[at(x, y)] = (std::int16_t)round; }

    // Separate from see() so the trainer's `mem` and `memfar` stay bit-identical.
    void saw_foe(int x, int y, int round) { foe[at(x, y)] = (std::int16_t)round; }
    void saw_ally(int x, int y, int round) { ally[at(x, y)] = (std::int16_t)round; }
    // Edges are terrain: whatever was seen on a side is what is there.
    void see_sides(int x, int y, std::uint8_t bits) { sides[at(x, y)] = bits; }

    // The grid's static-terrain (0-8) and memory (22-27) channels, into a
    // zeroed grid_cfg::N buffer. Inside the live window these read this turn's
    // values, since see() has already run for it.
    void grid(int hx, int hy, int facing, int round, float* out) const {
        using namespace grid_cfg;
        auto ch = [&](int c, int cell) -> float& { return out[(std::size_t)c * CELLS + cell]; };
        for (int row = 0; row < G; row++)
            for (int col = 0; col < G; col++) {
                int const cell = row * G + col;
                int wx, wy;
                mem_ego_to_world(facing, col - HALF, row - HALF, wx, wy);
                std::size_t const k = at(hx + wx, hy + wy);
                int const since = round - (int)visit[k];
                if (visit[k] > mem_cfg::NEVER) ch(VISITED, cell) = GridDecay::at(GRID_DECAY.visit, since);
                if (seen[k] <= mem_cfg::NEVER) continue;
                int const age = round - (int)seen[k];
                for (int d = 0; d < 4; d++) {          // ego side d is world side d + facing
                    int const wd = (d + facing) & 3;
                    // qualified: DragonMemory::KELP (a flag bit, 2) hides grid_cfg::KELP here
                    if (sides[k] >> wd & 1) ch(grid_cfg::KELP + d, cell) = 1.0f;
                    if (sides[k] >> (4 + wd) & 1) ch(grid_cfg::PORTAL + d, cell) = 1.0f;
                }
                if (cd[k] < 0) ch(NEVER_SPAWNS, cell) = 1.0f;
                ch(SEEN, cell) = GridDecay::at(GRID_DECAY.seen, age);
                if (expects_pearl(k, age)) ch(PEARL_EXPECTED, cell) = GridDecay::at(GRID_DECAY.pearl, age);
                if (cd[k] >= 0) ch(PEARL_TIMER_PROJ, cell) = (float)std::max((int)cd[k] - age, 0) / 99.0f;
                if (foe[k] > mem_cfg::NEVER) ch(ENEMY_MEM, cell) = GridDecay::at(GRID_DECAY.dragon, round - (int)foe[k]);
                if (ally[k] > mem_cfg::NEVER) ch(ALLY_MEM, cell) = GridDecay::at(GRID_DECAY.dragon, round - (int)ally[k]);
            }
    }

    bool expects_pearl(std::size_t k, int age) const {
        return (flags[k] & PEARL) != 0 || (cd[k] >= 0 && (int)cd[k] <= age);
    }

    // mem (676) then memfar (18), the order imitate2.py concatenated them in.
    void features(int hx, int hy, int facing, int round, float* out) const {
        mem(hx, hy, facing, round, out);
        memfar(hx, hy, facing, round, out + mem_cfg::N_MEM);
    }

    // The wide planes, channel-major: near's six 15x15 planes, then far's.
    // Laid out as a (2 * CH, SIDE, SIDE) tensor a conv stem can read directly.
    void wide(int hx, int hy, int facing, int round, float* out) const {
        using namespace wide_cfg;
        for (int row = 0; row < SIDE; row++)
            for (int col = 0; col < SIDE; col++) {
                int const i = row * SIDE + col;
                int const ox = col - R, oy = row - R;
                // near: one world cell per plane cell
                int wx, wy;
                mem_ego_to_world(facing, ox, oy, wx, wy);
                float near_ch[CH];
                cell(at(hx + wx, hy + wy), round, near_ch);
                for (int c = 0; c < CH; c++) out[(std::size_t)c * CELLS + i] = near_ch[c];

                // far: the POOL x POOL block of world cells centred on this
                // one, averaged. Blocks tile the plane, so the reach is
                // POOL * R + 1 rather than R.
                float acc[CH] = {};
                for (int sy = 0; sy < POOL; sy++)
                    for (int sx = 0; sx < POOL; sx++) {
                        int const ex = POOL * ox + sx - POOL / 2;
                        int const ey = POOL * oy + sy - POOL / 2;
                        int fx, fy;
                        mem_ego_to_world(facing, ex, ey, fx, fy);
                        float one[CH];
                        cell(at(hx + fx, hy + fy), round, one);
                        for (int c = 0; c < CH; c++) acc[c] += one[c];
                    }
                float const scale = 1.0f / (float)(POOL * POOL);
                for (int c = 0; c < CH; c++)
                    out[(std::size_t)(CH + c) * CELLS + i] = acc[c] * scale;
            }
    }

  private:
    // One remembered cell as wide_cfg::CH channels. The first four are `mem`'s,
    // read the same way, so the near scale is a superset of what the flat
    // feature carried; then the respawn timer and the decayed enemy sighting.
    void cell(std::size_t k, int round, float* out) const {
        using namespace mem_cfg;
        int const age = round - (int)seen[k];
        bool const known = age < UNKNOWN_AGE;
        out[0] = (known && expects_pearl(k, age)) ? 1.0f : 0.0f;
        out[1] = (known && (unsigned)age < (unsigned)AGE_MAX) ? MEM_DECAY.seen[age] : 0.0f;
        int const since = round - (int)visit[k];
        out[2] = (unsigned)since < (unsigned)VISIT_MAX ? MEM_DECAY.visit[since] : 0.0f;
        out[3] = (flags[k] & KELP) ? 1.0f : 0.0f;
        // 1 when a pearl is due now, falling to 0 at the 99-round cap; 0 where
        // the tile never spawns or was never seen.
        out[4] = (known && cd[k] >= 0) ? 1.0f - (float)std::min((int)cd[k], 99) / 99.0f : 0.0f;
        int const fage = round - (int)foe[k];
        out[5] = (fage < UNKNOWN_AGE && (unsigned)fage < (unsigned)AGE_MAX)
                     ? MEM_DECAY.seen[fage] : 0.0f;
    }

    void mem(int hx, int hy, int facing, int round, float* out) const {
        using namespace mem_cfg;
        float* expect_ch = out;
        float* seen_ch = out + CELLS;
        float* visit_ch = out + 2 * CELLS;
        float* kelp_ch = out + 3 * CELLS;
        auto const& ox = MEM_EGO.dx[(std::size_t)facing];
        auto const& oy = MEM_EGO.dy[(std::size_t)facing];
        for (int i = 0; i < CELLS; i++) {
            std::size_t const k = at(hx + ox[(std::size_t)i], hy + oy[(std::size_t)i]);
            int const age = round - (int)seen[k];
            bool const known = age < UNKNOWN_AGE;
            expect_ch[i] = (known && expects_pearl(k, age)) ? 1.0f : 0.0f;
            seen_ch[i] = (known && (unsigned)age < (unsigned)AGE_MAX) ? MEM_DECAY.seen[age] : 0.0f;
            int const since = round - (int)visit[k];
            visit_ch[i] = (unsigned)since < (unsigned)VISIT_MAX ? MEM_DECAY.visit[since] : 0.0f;
            kelp_ch[i] = (flags[k] & KELP) ? 1.0f : 0.0f;
        }
    }

    void memfar(int hx, int hy, int facing, int round, float* out) const {
        using namespace mem_cfg;
        for (int i = 0; i < N_FAR; i++) out[i] = 0.0f;
        int const halfw = w / 2, halfh = h / 2;
        int best_d[K], best_x[K], best_y[K];
        int found = 0, known = 0, pearls = 0;
        double cone[4] = {0, 0, 0, 0};          // forward, left, right, behind
        // row-major over (y, x): the order the offline builder's np.nonzero
        // gives, and therefore how ties on distance break
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                std::size_t const k = (std::size_t)y * (std::size_t)w + (std::size_t)x;
                int const s = (int)seen[k];
                if (s <= NEVER) continue;
                known++;
                if (!expects_pearl(k, round - s)) continue;
                pearls++;
                int const ddx = mem_wrap(x - hx + halfw, w) - halfw;
                int const ddy = mem_wrap(y - hy + halfh, h) - halfh;
                int ex = 0, ey = 0;
                switch (facing) {
                    case 0: ex = ddx;  ey = ddy;  break;
                    case 1: ex = ddy;  ey = -ddx; break;
                    case 2: ex = -ddx; ey = -ddy; break;
                    default: ex = -ddy; ey = ddx; break;
                }
                int const ax = ex < 0 ? -ex : ex, ay = ey < 0 ? -ey : ey;
                int const dist = ax + ay;
                if (found < K) { best_d[found] = 1 << 30; found++; }
                for (int j = 0; j < found; j++)      // strictly-less keeps ties stable
                    if (dist < best_d[j]) {
                        for (int q = found - 1; q > j; q--) {
                            best_d[q] = best_d[q - 1];
                            best_x[q] = best_x[q - 1];
                            best_y[q] = best_y[q - 1];
                        }
                        best_d[j] = dist;
                        best_x[j] = ex;
                        best_y[j] = ey;
                        break;
                    }
                double const wgt = 1.0 / (1.0 + (double)dist);
                if (ey < 0 && -ey >= ax) cone[0] += wgt;
                if (-ex > ay) cone[1] += wgt;
                if (ex > ay) cone[2] += wgt;
                if (ey > 0 && ey >= ax) cone[3] += wgt;
            }
        for (int j = 0; j < K; j++) {
            if (j >= found || best_d[j] >= (1 << 30)) continue;
            out[4 * j] = 1.0f;
            out[4 * j + 1] = mem_half((float)((double)best_x[j] / 16.0));
            out[4 * j + 2] = mem_half((float)((double)best_y[j] / 16.0));
            out[4 * j + 3] = mem_half((float)((double)best_d[j] / 32.0));
        }
        for (int j = 0; j < 4; j++)
            out[4 * K + j] = mem_half((float)((cone[j] < 4.0 ? cone[j] : 4.0) / 4.0));
        out[4 * K + 4] = mem_half((float)((double)known / ((double)w * (double)h)));
        out[4 * K + 5] = mem_half((float)((pearls < 32 ? pearls : 32) / 32.0));
    }
};

}  // namespace bc
