// Reward v8: one bounded zero-sum team potential. See REWARDS.md for the
// derivation and for the measurements behind the constants here.
//
// Kept in its own header, free of Game and Env, so the whole thing is a pure
// function of two teams' shapes and can be tested without an engine. bc_vec.hpp
// supplies the shapes and does the banking.
//
// The contract, and the three things that are easy to get wrong:
//
//  * Every component is antisymmetric under swapping the teams, so
//    Phi(swap) == -Phi exactly and self-play stays zero-sum. The tests assert
//    this over random shapes; it is the cheapest check that the whole design
//    still holds.
//  * The components are returned ALREADY scaled by kappa * lambda_i / sum
//    (lambda). The caller banks these numbers, not the bare Phi_i: the weights
//    and the normaliser are both functions of the round, and banking anything
//    less than the fully evaluated value reintroduces a reward proportional to
//    -Phi * d(lambda)/dt that pays the policy not to be ahead early.
//  * The normalisation is load-bearing, not cosmetic: |Phi| <= kappa, so the
//    game's final Phi is a bounded score.
//
// The queen rules (unswbc 1.2.3, 2026-10-01): after the last round the winner is
// the team with the longer queen (dragon id 0 or 1, one a team, dead = length
// 0), then the longest living dragon, then the total length. WIN reads them in
// that order; QUEEN prices a queen being alive at all, flat until the last 125
// rounds and then fading out, so the end is WIN alone and a late queen kill is
// not paid twice (user, 2026-10-01: no reckless late swings at the queen).
#pragma once

#include <algorithm>
#include <cmath>

namespace bc8 {

// Which components exist. Banked and logged separately even though they sum to
// one scalar: it costs nothing and it is what preserves the per-term dashboard
// attribution that caught reward v3's kamikaze collapse inside 30 iterations.
enum Term { T_WIN = 0, T_LEN, T_QUEEN, T_KILL, T_EXP, N_TERMS };

constexpr int QUEENS_PER_TEAM = 1;   // a team's queens at the start (the rules: exactly one)

// What the potential reads off one team. Living dragons only.
struct TeamShape {
    int total = 0;      // sum of lengths, the last tie-break
    int longest = 0;    // the second tie-break
    int units = 0;      // living dragons
    int queen = 0;      // the queen's length, 0 once it is dead: the first tie-break
    int queens = 0;     // living queens (0 or 1)
    int covered = 0;    // distinct tiles this team's heads have stood on
};

struct Params {
    float kappa = 1.0f;      // shaping strength against the outcome; the one knob
    float lambda_in = 0.4f;  // weight of each inner tie-break inside T_WIN (see tiebreak_gain)
    float kill_c = 15.0f;    // finisher length scale, in segments (see raw_terms)
    float eps = 0.005f;      // per fresh tile of coverage LEAD
};

struct Lambdas {
    float v[N_TERMS] = {0};
    float sum = 0.0f;
};

inline float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }

// Hand-set schedules of the round. EXP also decays with how much of the map is
// known, so it scales itself to map size, and is gone by round 75 (2026-10-01:
// it predicts the winner early but adds nothing once lengths are known).
inline Lambdas lambdas(int round, int max_rounds, int covered_max, int area) {
    const float s = max_rounds > 0 ? clamp01((float)round / (float)max_rounds) : 0.0f;
    const float known = area > 0 ? clamp01((float)covered_max / (float)area) : 1.0f;
    Lambdas L;
    // 0 at the start (user, 2026-10-01: early on a queen raising children beats racing for length)
    L.v[T_WIN] = 1.5f * s * s;                                           // 0 -> 1.50
    L.v[T_LEN] = 1.0f - s;                                               // 1.00 -> 0
    L.v[T_QUEEN] = clamp01((float)(max_rounds - round) / 125.0f);        // 1 to round 375, 0 at 500
    L.v[T_KILL] = 0.8f * (1.0f - s * s * s);                             // 0.80 -> 0
    L.v[T_EXP] = (1.0f - known) * clamp01((75.0f - (float)round) / 50.0f);
    for (int i = 0; i < N_TERMS; i++) L.sum += L.v[i];
    return L;
}

// 2(a-b)/(a+b), the normalised difference every material term is built from.
// Bounded to [-2, 2] by construction; 0/0 is 0 -- which for the queens is the
// rules exactly: both queens dead and the verdict falls to the longest dragon.
inline float norm_diff(int a, int b) {
    const int denom = a + b;
    if (denom <= 0) return 0.0f;
    const float v = 2.0f * (float)(a - b) / (float)denom;
    return std::min(2.0f, std::max(-2.0f, v));
}

// How much the next tie-break may move a level whose two quantities are a and b.
// Lengths are integers, so a nonzero norm_diff(a, b) is at least 2 / (a + b) >=
// 1 / max(a, b): scaled by this (and lambda_in < 1) an inner level can never
// outvote a real gap, so WIN orders exactly as the verdict does. Both zero (both
// queens dead) hands the decision down whole; one zero has already decided it.
inline float tiebreak_gain(int a, int b) {
    if (a <= 0 && b <= 0) return 1.0f;
    if (a <= 0 || b <= 0) return 0.0f;
    return 1.0f / (float)std::max(a, b);
}

// The raw, unweighted components, each in [-1, 1] and each antisymmetric.
inline void raw_terms(const TeamShape& us, const TeamShape& them, const Params& p, float out[N_TERMS]) {
    const float q = norm_diff(us.queen, them.queen);
    const float r = norm_diff(us.longest, them.longest);
    const float z = norm_diff(us.total, them.total);

    // The round-500 verdict, nested in its own order, each inner level scaled so it
    // cannot outvote the one above (tiebreak_gain). An unscaled nesting -- or an added
    // longest-dragon term -- trades against the queen: queens 20 v 21 with longest
    // 40 v 21 scored as winning (0.6% of random endings had the wrong sign).
    const float gq = p.lambda_in * tiebreak_gain(us.queen, them.queen);
    const float gr = p.lambda_in * tiebreak_gain(us.longest, them.longest);
    out[T_WIN] = std::tanh(q + gq * std::tanh(r + gr * std::tanh(z)));
    out[T_LEN] = std::tanh(z);
    out[T_QUEEN] = (float)(us.queens - them.queens) / (float)QUEENS_PER_TEAM;

    // "How close are they to being wiped out", convex so the last kills are
    // worth the most and elimination -- which still ends the game outright -- is
    // the maximum. Over TOTAL LENGTH, not unit count: a team is eliminated
    // exactly when its total reaches zero, and a split conserves length where a
    // count would pay for splitting (d/dN [-exp(-N/c)] > 0).
    out[T_KILL] = std::exp(-(float)them.total / p.kill_c) -
                  std::exp(-(float)us.total / p.kill_c);

    // A coverage LEAD, not coverage: this has to stay antisymmetric. eps is
    // per-tile rather than per-unit-area so that a fresh tile is worth the same
    // on arena and on big_empty; tanh bounds the term once the lead is large.
    out[T_EXP] = std::tanh(p.eps * (float)(us.covered - them.covered));
}

// The banked value: kappa * lambda_i(t) * Phi_i / sum(lambda(t)), per component.
// Summing these gives Phi, with |Phi| <= kappa.
inline void potential(const TeamShape& us, const TeamShape& them, int round, int max_rounds,
                      int area, const Params& p, float out[N_TERMS]) {
    float raw[N_TERMS];
    raw_terms(us, them, p, raw);
    const Lambdas L = lambdas(round, max_rounds, std::max(us.covered, them.covered), area);
    const float scale = L.sum > 1e-6f ? p.kappa / L.sum : 0.0f;
    for (int i = 0; i < N_TERMS; i++) out[i] = scale * L.v[i] * raw[i];
}

// Convenience for tests and asserts.
inline float potential_sum(const TeamShape& us, const TeamShape& them, int round,
                           int max_rounds, int area, const Params& p) {
    float t[N_TERMS];
    potential(us, them, round, max_rounds, area, p, t);
    float s = 0.0f;
    for (int i = 0; i < N_TERMS; i++) s += t[i];
    return s;
}

}  // namespace bc8
