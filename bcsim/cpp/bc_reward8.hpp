// Reward v8: one bounded zero-sum team potential. See REWARDS.md for the
// derivation and for the measurements behind every constant here.
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
//  * The normalisation is load-bearing, not cosmetic. Un-normalised, a crushing
//    win is charged -1.02 for ending the game, because the terminal payment is
//    W * outcome - Phi(s_T) and a raw Phi overshoots W. With |Phi| <= kappa = W
//    the payment for winning is non-negative by construction.
#pragma once

#include <algorithm>
#include <cmath>

namespace bc8 {

// Which components exist. Banked and logged separately even though they sum to
// one scalar: it costs nothing and it is what preserves the per-term dashboard
// attribution that caught reward v3's kamikaze collapse inside 30 iterations.
enum Term { T_WIN = 0, T_LEN, T_TOP3, T_KILL, T_EXP, N_TERMS };

// What the potential reads off one team. Living dragons only.
struct TeamShape {
    int total = 0;      // sum of lengths, the real tie-break
    int longest = 0;    // the win condition
    int units = 0;      // living dragons
    int top3 = 0;       // sum of the three longest, "eggs in more than one basket"
    int covered = 0;    // distinct tiles this team's heads have stood on
};

struct Params {
    float kappa = 1.0f;      // shaping strength against the outcome; the one knob
    float lambda_z = 0.4f;   // weight of the total-length tie-break inside T_WIN
    float kill_c = 15.0f;    // finisher length scale, in segments (see raw_terms)
    float eps = 0.005f;      // per fresh tile of coverage LEAD
    float outcome_w = 1.0f;  // W: the only non-telescoping term
};

struct Lambdas {
    float v[N_TERMS] = {0};
    float sum = 0.0f;
};

// Hand-set schedules. Constants of the round, except lambda_exp, which decays
// with how much of the map is known so that it scales itself to map size: a
// clock at round 150 is spent before arena is half over and still leaves 87% of
// big_empty unexplored.
inline Lambdas lambdas(int round, int max_rounds, int covered_max, int area) {
    const float s = max_rounds > 0
                        ? std::min(1.0f, std::max(0.0f, (float)round / (float)max_rounds))
                        : 0.0f;
    const float known = area > 0
                            ? std::min(1.0f, std::max(0.0f, (float)covered_max / (float)area))
                            : 1.0f;
    Lambdas L;
    L.v[T_WIN] = 0.3f + 1.2f * s * s;                                  // 0.30 -> 1.50
    L.v[T_LEN] = 0.2f + 0.8f * (1.0f - s);                             // 1.00 -> 0.20
    L.v[T_TOP3] = 0.6f * std::min(1.0f, std::max(0.0f, (s - 0.4f) / 0.6f));
    L.v[T_KILL] = 0.8f * (1.0f - s * s * s);                           // 0.80 -> 0
    L.v[T_EXP] = 1.0f - known;                                         // 1.00 -> 0
    for (int i = 0; i < N_TERMS; i++) L.sum += L.v[i];
    return L;
}

// 2(a-b)/(a+b), the normalised difference every material term is built from.
// Bounded to [-2, 2] by construction; 0/0 is 0, which only arises once both
// teams are wiped and the game is already over.
inline float norm_diff(int a, int b) {
    const int denom = a + b;
    if (denom <= 0) return 0.0f;
    const float v = 2.0f * (float)(a - b) / (float)denom;
    return std::min(2.0f, std::max(-2.0f, v));
}

// The raw, unweighted components, each in [-1, 1] and each antisymmetric.
inline void raw_terms(const TeamShape& us, const TeamShape& them, const Params& p,
                      float out[N_TERMS]) {
    const float r = norm_diff(us.longest, them.longest);
    const float z = norm_diff(us.total, them.total);
    const float a = norm_diff(us.top3, them.top3);

    // The closed form, not tanh(r) + lambda_z * sech^2(r) * tanh(z). That
    // additive version is its first-order expansion (sech^2 is tanh'), and it
    // goes non-monotone in r once 2 * lambda_z * tanh(r) * tanh(z) > 1 -- a
    // region where growing your leader lowers your reward. This cannot.
    out[T_WIN] = std::tanh(r + p.lambda_z * std::tanh(z));
    out[T_LEN] = std::tanh(z);
    out[T_TOP3] = std::tanh(a);

    // "How close are they to being wiped out", convex so the last kills are
    // worth the most and elimination is the maximum.
    //
    // Measured over TOTAL LENGTH, not unit count. A team is eliminated exactly
    // when its total reaches zero, so length says the same thing about
    // elimination -- and a split conserves it, where unit count does not.
    // Antisymmetry forces a matching term in our own quantity, and with counts
    // that term pays us for splitting: d/dN_0 [-exp(-N_0/c)] > 0 always, worth
    // +0.019 for a free split of a non-leader, which is v0's shredding rebuilt
    // at a fifth the size. Over length the same derivative pays us for being
    // longer, which is a direction we want anyway.
    //
    // Rejected: (N_E * tanh(N_0/c) - N_0 * tanh(N_E/c)) / (N_0 + N_E), which is
    // negative for EVERY advantage at every c -- tanh(x)/x is decreasing -- and
    // is zero at the moment of victory.
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
