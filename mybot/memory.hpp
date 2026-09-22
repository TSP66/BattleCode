// What this dragon remembers, as the network's extra inputs.
//
// The simulator's observation is one 7x7 window and 14 scalars, and a clone
// trained on that alone cannot predict a team that plays off a remembered map.
// bcsim/train/clone_features.py adds two inputs computed from the dragon's own
// earlier turns -- `mem` (676: the 13x13 around the head, in its own frame)
// and `memfar` (18: the whole remembered map, summarised) -- and they are
// worth +2.9 points of held-out accuracy and +0.03 of win rate (see
// DISTILL_DEVTEST.md).
//
// A deployed dragon can compute them because the judge runs one process per
// dragon: init() reads its ID once and update() then loops over that dragon's
// turns, so one instance of this class lives exactly as long as the dragon.
// A split child is a new process and therefore starts with an empty memory,
// which is what the trainer assumed ("a split child is a new process").
//
// Every value here has to equal clone_features.MemoryTracker's, because the
// checkpoint was chosen by win rates measured through that tracker; a
// disagreement feeds the network inputs it never saw. wasmprobe/parity_mem.py
// is the check, and it compares exactly, not approximately:
//
//   * the decayed ages are quantised to a byte offline (uint8, /255), so they
//     come out of a table built with the same rounding rather than from expf;
//   * memfar is stored as float16, so each of its 18 values is rounded to
//     half precision before the network sees it.
#pragma once

#include "obs.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace memory {

constexpr int R = 6;                            // mem is the 13x13 around the head
constexpr int SIDE = 2 * R + 1;
constexpr int CELLS = SIDE * SIDE;              // 169
constexpr int N_MEM = 4 * CELLS;                // 676
constexpr int K = 3;                            // the nearest expected pearls memfar names
constexpr int N_FAR = 4 * K + 6;                // 18
constexpr int N_EXTRA = N_MEM + N_FAR;          // 694
constexpr int N_SCALARS_IN = obs::N_SCALARS + N_EXTRA;   // 708, the net's scalar input

constexpr int NEVER = -10'000;                  // "not seen", as the builders write it
// 255 * exp(-age/32) and 255 * exp(-age/16) round to zero past these
constexpr int AGE_MAX = 200;
constexpr int VISIT_MAX = 100;

inline int wrap(int v, int m) { return (v % m + m) % m; }

// ego (row, col) of the 13x13 -> world offset, per facing. The same rotation
// obs::Tables::perm applies to the window, at radius 6 instead of 3.
struct EgoTable {
    std::array<std::array<std::int8_t, CELLS>, 4> dx{}, dy{};

    constexpr EgoTable() {
        for (int f = 0; f < 4; f++)
            for (int row = 0; row < SIDE; row++)
                for (int col = 0; col < SIDE; col++) {
                    int const ox = col - R, oy = row - R;
                    int wx = 0, wy = 0;
                    switch (f) {
                        case 0: wx = ox;  wy = oy;  break;
                        case 1: wx = -oy; wy = ox;  break;
                        case 2: wx = -ox; wy = -oy; break;
                        default: wx = oy; wy = -ox; break;
                    }
                    dx[f][(std::size_t)(row * SIDE + col)] = (std::int8_t)wx;
                    dy[f][(std::size_t)(row * SIDE + col)] = (std::int8_t)wy;
                }
    }
};

inline constexpr EgoTable EGO{};

// Rounds to what float16 can hold, which is how memfar was stored for
// training and what MemoryTracker feeds the policy at evaluation time.
// Round to nearest, ties to even, on the top 10 mantissa bits. Every memfar
// value is 0 or between 2^-12 and 4, so half precision never needs a
// subnormal or overflows here.
inline float half(float x) {
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

class Memory {
  public:
    // Sized from the map, because memfar walks every remembered cell. 64x64
    // is the largest map, so this is at most 57 KB.
    void init(int w, int h) {
        w_ = w;
        h_ = h;
        std::size_t const n = (std::size_t)w * (std::size_t)h;
        seen_.assign(n, NEVER);
        visit_.assign(n, NEVER);
        cd_.assign(n, -1);
        pearl_.assign(n, 0);
        kelp_.assign(n, 0);
        for (int a = 0; a < AGE_MAX; a++)
            seen_decay_[a] = (float)(std::nearbyint(255.0 * std::exp(-(double)a / 32.0)) / 255.0);
        for (int a = 0; a < VISIT_MAX; a++)
            visit_decay_[a] = (float)(std::nearbyint(255.0 * std::exp(-(double)a / 16.0)) / 255.0);
    }

    // This turn's window, then where the head stands. Called on every turn,
    // including the first and any turn the network does not get to run on: a
    // gap would change every feature after it.
    void observe(obs::Snapshot const& snap, int round) {
        for (int i = 0; i < obs::CELLS; i++) {
            auto const& t = snap.tile(i);
            std::size_t const k = at(t.position.x, t.position.y);
            seen_[k] = round;
            pearl_[k] = t.pearl ? 1 : 0;
            // the observation caps the timer at 99 and flags "never" as -1;
            // the memory keeps what the observation showed, not the truth
            cd_[k] = t.pearl_time < 0 ? -1 : (t.pearl_time < 99 ? t.pearl_time : 99);
            for (int d = 0; d < 4; d++)
                if (t.edges[(std::size_t)d].edge_type == unswbc::EdgeType::KELP) {
                    kelp_[k] = 1;       // kelp is remembered once seen, never cleared
                    break;
                }
        }
        visit_[at(snap.hx, snap.hy)] = round;
    }

    // mem (676) then memfar (18), in the order imitate2.py concatenated the
    // features the checkpoint names.
    void features(obs::Snapshot const& snap, int round, float* out) const {
        mem(snap, round, out);
        memfar(snap, round, out + N_MEM);
    }

  private:
    int w_ = 1, h_ = 1;
    std::vector<std::int32_t> seen_, visit_, cd_;
    std::vector<std::uint8_t> pearl_, kelp_;
    float seen_decay_[AGE_MAX]{};
    float visit_decay_[VISIT_MAX]{};

    std::size_t at(int x, int y) const {
        return (std::size_t)wrap(y, h_) * (std::size_t)w_ + (std::size_t)wrap(x, w_);
    }

    // The 13x13 in the dragon's own frame: pearls it expects there (seen, or a
    // respawn timer that has run out), how recently it saw the cell, where its
    // own head has been, and kelp it has seen.
    void mem(obs::Snapshot const& snap, int round, float* out) const {
        float* expect_ch = out;
        float* seen_ch = out + CELLS;
        float* visit_ch = out + 2 * CELLS;
        float* kelp_ch = out + 3 * CELLS;
        auto const& ox = EGO.dx[(std::size_t)snap.facing];
        auto const& oy = EGO.dy[(std::size_t)snap.facing];
        for (int i = 0; i < CELLS; i++) {
            std::size_t const k = at(snap.hx + ox[(std::size_t)i], snap.hy + oy[(std::size_t)i]);
            int const age = round - seen_[k];
            bool const known = age < 10'000;
            bool const expect = known && (pearl_[k] != 0 || (cd_[k] >= 0 && cd_[k] <= age));
            expect_ch[i] = expect ? 1.0f : 0.0f;
            seen_ch[i] = known && (unsigned)age < (unsigned)AGE_MAX
                         ? seen_decay_[age] : 0.0f;
            int const since = round - visit_[k];
            visit_ch[i] = (unsigned)since < (unsigned)VISIT_MAX ? visit_decay_[since] : 0.0f;
            kelp_ch[i] = kelp_[k] != 0 ? 1.0f : 0.0f;
        }
    }

    // The whole remembered map, summarised: the K nearest expected pearls
    // (present, ego dx, dy, distance), the 1/(1+d)-weighted count of expected
    // pearls ahead / left / right / behind, the share of the map seen, and how
    // many pearls are expected anywhere.
    void memfar(obs::Snapshot const& snap, int round, float* out) const {
        for (int i = 0; i < N_FAR; i++) out[i] = 0.0f;
        int const f = snap.facing;
        int const halfw = w_ / 2, halfh = h_ / 2;
        int best_d[K], best_x[K], best_y[K];
        int found = 0, known = 0, pearls = 0;
        double cone[4] = {0, 0, 0, 0};          // forward, left, right, behind
        // row-major over (y, x), which is the order the offline builder's
        // np.nonzero gives and therefore how ties on distance break
        for (int y = 0; y < h_; y++)
            for (int x = 0; x < w_; x++) {
                std::size_t const k = (std::size_t)y * (std::size_t)w_ + (std::size_t)x;
                int const seen = seen_[k];
                if (seen <= NEVER) continue;
                known++;
                if (pearl_[k] == 0 && !(cd_[k] >= 0 && cd_[k] <= round - seen)) continue;
                pearls++;
                int const ddx = wrap(x - snap.hx + halfw, w_) - halfw;
                int const ddy = wrap(y - snap.hy + halfh, h_) - halfh;
                int ex = 0, ey = 0;
                switch (f) {
                    case 0: ex = ddx;  ey = ddy;  break;
                    case 1: ex = ddy;  ey = -ddx; break;
                    case 2: ex = -ddx; ey = -ddy; break;
                    default: ex = -ddy; ey = ddx; break;
                }
                int const ax = ex < 0 ? -ex : ex, ay = ey < 0 ? -ey : ey;
                int const dist = ax + ay;
                if (found < K) { best_d[found] = 1 << 30; found++; }
                for (int j = 0; j < found; j++)          // strictly-less keeps ties stable
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
            out[4 * j + 1] = half((float)((double)best_x[j] / 16.0));
            out[4 * j + 2] = half((float)((double)best_y[j] / 16.0));
            out[4 * j + 3] = half((float)((double)best_d[j] / 32.0));
        }
        for (int j = 0; j < 4; j++)
            out[4 * K + j] = half((float)((cone[j] < 4.0 ? cone[j] : 4.0) / 4.0));
        out[4 * K + 4] = half((float)((double)known / ((double)w_ * (double)h_)));
        out[4 * K + 5] = half((float)((pearls < 32 ? pearls : 32) / 32.0));
    }
};

}  // namespace memory
