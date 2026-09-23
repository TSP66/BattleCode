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

    static constexpr std::uint8_t PEARL = 1, KELP = 2;

    void reset(int w_, int h_) {
        w = w_;
        h = h_;
        std::size_t const n = (std::size_t)w * (std::size_t)h;
        seen.assign(n, (std::int16_t)mem_cfg::NEVER);
        visit.assign(n, (std::int16_t)mem_cfg::NEVER);
        cd.assign(n, (std::int8_t)-1);
        flags.assign(n, 0);
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

    bool expects_pearl(std::size_t k, int age) const {
        return (flags[k] & PEARL) != 0 || (cd[k] >= 0 && (int)cd[k] <= age);
    }

    // mem (676) then memfar (18), the order imitate2.py concatenated them in.
    void features(int hx, int hy, int facing, int round, float* out) const {
        mem(hx, hy, facing, round, out);
        memfar(hx, hy, facing, round, out + mem_cfg::N_MEM);
    }

  private:
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
