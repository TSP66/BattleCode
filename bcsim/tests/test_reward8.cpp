// Asserts the properties REWARDS.md claims for reward v8, over random team
// shapes. Antisymmetry and boundedness are the two that the whole design rests
// on; the rest are the specific mistakes the draft made, kept as regressions so
// they cannot come back.
//
//   g++ -O2 -std=c++17 -I bcsim/cpp bcsim/tests/test_reward8.cpp -o /tmp/t8 && /tmp/t8
#include "bc_reward8.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace bc8;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("  FAIL  %s\n", what);
        failures++;
    }
}

static TeamShape shape(std::vector<int> lens, int covered) {
    TeamShape t;
    std::sort(lens.begin(), lens.end(), std::greater<int>());
    for (int l : lens) {
        t.total += l;
        t.units++;
    }
    t.longest = lens.empty() ? 0 : lens[0];
    for (size_t i = 0; i < lens.size() && i < 3; i++) t.top3 += lens[i];
    t.covered = covered;
    return t;
}

int main() {
    Params p;
    const int MAXR = 500;

    // ---- antisymmetry and boundedness, over random shapes and rounds
    std::mt19937 rng(12345);
    float worst_anti = 0.0f, worst_abs = 0.0f;
    for (int trial = 0; trial < 200000; trial++) {
        const int n0 = (int)(rng() % 12), nE = (int)(rng() % 12);
        std::vector<int> a, b;
        for (int i = 0; i < n0; i++) a.push_back(1 + (int)(rng() % 40));
        for (int i = 0; i < nE; i++) b.push_back(1 + (int)(rng() % 40));
        const int area = 121 + (int)(rng() % 4000);
        const TeamShape us = shape(a, (int)(rng() % (area + 1)));
        const TeamShape them = shape(b, (int)(rng() % (area + 1)));
        const int round = (int)(rng() % (MAXR + 1));

        float f[N_TERMS], g[N_TERMS];
        potential(us, them, round, MAXR, area, p, f);
        potential(them, us, round, MAXR, area, p, g);
        float sf = 0, sg = 0;
        for (int i = 0; i < N_TERMS; i++) {
            worst_anti = std::max(worst_anti, std::fabs(f[i] + g[i]));
            sf += f[i];
            sg += g[i];
        }
        worst_abs = std::max(worst_abs, std::fabs(sf));
        check(std::fabs(sf + sg) < 1e-5f, "sum antisymmetry");
    }
    std::printf("antisymmetry: worst |Phi_i(us,them) + Phi_i(them,us)| = %.3e\n", worst_anti);
    std::printf("boundedness:  worst |Phi| = %.4f   (must be <= kappa = %.2f)\n",
                worst_abs, p.kappa);
    check(worst_anti < 1e-5f, "per-term antisymmetry");
    check(worst_abs <= p.kappa + 1e-5f, "|Phi| <= kappa");

    // ---- identical teams are exactly zero, so a symmetric spawn has Phi = 0
    // and the whole episode's shaped return telescopes to nothing.
    for (int round : {0, 1, 137, 250, 499, 500}) {
        const TeamShape s = shape({8, 6, 5, 4}, 300);
        const float v = potential_sum(s, s, round, MAXR, 1024, p);
        check(std::fabs(v) < 1e-6f, "identical teams give Phi = 0");
        (void)v;
    }

    // ---- winning is never punished at the terminal: R_T = W * outcome - Phi
    // must be >= 0 for a win. This is the bug the normalisation fixes; raw Phi
    // charged a crushing win -1.02 for ending the game.
    float worst_win_payment = 1e9f;
    for (int trial = 0; trial < 200000; trial++) {
        std::vector<int> a, b;
        const int n0 = 1 + (int)(rng() % 10), nE = 1 + (int)(rng() % 10);
        for (int i = 0; i < n0; i++) a.push_back(1 + (int)(rng() % 60));
        for (int i = 0; i < nE; i++) b.push_back(1 + (int)(rng() % 60));
        const int area = 121 + (int)(rng() % 4000);
        const TeamShape us = shape(a, (int)(rng() % (area + 1)));
        const TeamShape them = shape(b, (int)(rng() % (area + 1)));
        const float phi = potential_sum(us, them, MAXR, MAXR, area, p);
        worst_win_payment = std::min(worst_win_payment, p.outcome_w - phi);
    }
    std::printf("terminal:     worst payment for a WIN = %+.4f  (must be >= 0)\n",
                worst_win_payment);
    check(worst_win_payment >= -1e-5f, "a win is never charged");

    // ---- the finisher must rise monotonically as the enemy's material falls,
    // at every scale, and be maximal at elimination. The draft's concave form
    // was negative for every advantage we tried.
    for (float c : {5.0f, 15.0f, 40.0f}) {
        Params q = p;
        q.kill_c = c;
        float prev = -2.0f;
        for (int their_total = 100; their_total >= 0; their_total -= 5) {
            float t[N_TERMS];
            raw_terms(shape({20, 10, 5}, 0),
                      their_total > 0 ? shape({their_total}, 0) : shape({}, 0), q, t);
            check(t[T_KILL] > prev, "finisher rises as their material falls");
            prev = t[T_KILL];
        }
        float wipe[N_TERMS], parity[N_TERMS];
        raw_terms(shape({20, 10, 5}, 0), shape({}, 0), q, wipe);
        raw_terms(shape({20, 10, 5}, 0), shape({20, 10, 5}, 0), q, parity);
        check(wipe[T_KILL] > 0.0f, "finisher is positive at a wipe-out");
        check(std::fabs(parity[T_KILL]) < 1e-6f, "finisher is zero at parity");
    }

    // ---- and it must be EXACTLY split-invariant, which is why it is written
    // over total length rather than unit count.
    for (float c : {5.0f, 15.0f, 40.0f}) {
        Params q = p;
        q.kill_c = c;
        float before[N_TERMS], after[N_TERMS];
        raw_terms(shape({20, 14, 9, 6, 4}, 0), shape({22, 15}, 0), q, before);
        raw_terms(shape({20, 14, 9, 6, 2, 2}, 0), shape({22, 15}, 0), q, after);
        check(std::fabs(after[T_KILL] - before[T_KILL]) < 1e-6f,
              "finisher is split-invariant");
    }

    // ---- T_WIN monotone in our longest, which the draft's additive form was
    // not for lambda_z > 0.5.
    for (float lz : {0.2f, 0.4f, 0.5f, 0.9f}) {
        Params q = p;
        q.lambda_z = lz;
        for (int their_total : {10, 60, 200}) {
            float prev = -2.0f;
            bool mono = true;
            for (int ours = 1; ours <= 60; ours++) {
                float t[N_TERMS];
                raw_terms(shape({ours}, 0), shape({20, their_total / 2}, 0), q, t);
                if (t[T_WIN] < prev - 1e-6f) mono = false;
                prev = t[T_WIN];
            }
            check(mono, "T_WIN monotone in our longest");
        }
    }

    // ---- a free split of a NON-leader must not pay. This is v0's shredding,
    // which the draft's count term rebuilt (+0.147 to +0.016 everywhere).
    for (int round : {0, 100, 250, 400, 500}) {
        const TeamShape before = shape({20, 14, 9, 6, 4}, 400);
        const TeamShape after = shape({20, 14, 9, 6, 2, 2}, 400);   // split the 4
        const TeamShape foe = shape({22, 15, 10, 5, 3}, 400);
        const float d = potential_sum(after, foe, round, MAXR, 1024, p) -
                        potential_sum(before, foe, round, MAXR, 1024, p);
        std::printf("split a non-leader at r%-3d  dPhi %+.5f\n", round, d);
        check(d <= 1e-6f, "a free split of a non-leader does not pay");
    }

    // ---- killing their biggest must beat killing their smallest, by a lot,
    // and a good trade must beat an even one.
    {
        const TeamShape us = shape({20, 14, 9, 6, 4}, 400);
        const TeamShape foe = shape({22, 15, 10, 5, 3}, 400);
        const float base = potential_sum(us, foe, 250, MAXR, 1024, p);
        const float pearl = potential_sum(shape({20, 14, 9, 6, 5}, 400), foe, 250, MAXR, 1024, p) - base;
        const float small = potential_sum(us, shape({22, 15, 10, 5}, 400), 250, MAXR, 1024, p) - base;
        const float big = potential_sum(us, shape({15, 10, 5, 3}, 400), 250, MAXR, 1024, p) - base;
        const float sac = potential_sum(shape({20, 14, 9, 6}, 400),
                                        shape({15, 10, 5, 3}, 400), 250, MAXR, 1024, p) - base;
        const float even = potential_sum(shape({14, 9, 6, 4}, 400),
                                         shape({15, 10, 5, 3}, 400), 250, MAXR, 1024, p) - base;
        std::printf("pearl %+.4f  kill-small %+.4f  kill-big %+.4f  sac-4-for-22 %+.4f  "
                    "leader-trade %+.4f\n", pearl, small, big, sac, even);
        check(pearl > 0, "a pearl pays");
        check(big > 4 * small, "killing their leader beats killing their smallest");
        check(big > 20 * pearl, "killing their leader dwarfs a pearl");
        check(sac > 10 * pearl, "trading a small dragon for their leader pays well");
        check(sac > even, "a good trade beats an even one");
        check(even > 0, "trading up, even narrowly, still pays");
    }

    // ---- coverage: a fresh tile pays while the map is unknown, and nothing
    // once it is known.
    {
        const TeamShape foe = shape({8, 6}, 100);
        const float open_gain =
            potential_sum(shape({8, 6}, 105), foe, 20, MAXR, 1024, p) -
            potential_sum(shape({8, 6}, 100), foe, 20, MAXR, 1024, p);
        const TeamShape foe_done = shape({8, 6}, 1024);
        const float done_gain =
            potential_sum(shape({8, 6}, 1024), foe_done, 20, MAXR, 1024, p) -
            potential_sum(shape({8, 6}, 1019), foe_done, 20, MAXR, 1024, p);
        std::printf("5 fresh tiles: unexplored map %+.5f   fully explored %+.5f\n",
                    open_gain, done_gain);
        check(open_gain > 0, "coverage pays while the map is unknown");
        check(std::fabs(done_gain) < std::fabs(open_gain) * 0.05f,
              "coverage stops paying once the map is known");
    }

    std::printf(failures ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
