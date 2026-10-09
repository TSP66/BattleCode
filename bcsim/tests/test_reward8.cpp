// Asserts the properties REWARDS.md claims for reward v8 (queen form, 2026-10-01),
// over random team shapes. Antisymmetry and boundedness are the two that the
// whole design rests on; the rest are the specific mistakes earlier drafts made,
// kept as regressions so they cannot come back, and the queen rules.
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

// lens[queen] is the queen; queen < 0 = the queen is dead
static TeamShape shape(std::vector<int> lens, int covered, int queen = 0) {
    TeamShape t;
    if (queen >= 0 && queen < (int)lens.size()) {
        t.queen = lens[(size_t)queen];
        t.queens = 1;
    }
    for (int l : lens) {
        t.total += l;
        t.units++;
        t.longest = std::max(t.longest, l);
    }
    t.covered = covered;
    return t;
}

static TeamShape random_shape(std::mt19937& rng, int max_len, int area) {
    std::vector<int> lens;
    for (int i = 0, n = (int)(rng() % 12); i < n; i++) lens.push_back(2 + (int)(rng() % (max_len - 1)));
    const int queen = lens.empty() || rng() % 4 == 0 ? -1 : (int)(rng() % lens.size());
    return shape(lens, (int)(rng() % (area + 1)), queen);
}

// the round-500 verdict: +1 we win, -1 they win, 0 draw (both alive: the rules' order)
static int verdict(const TeamShape& a, const TeamShape& b) {
    if (a.units == 0 || b.units == 0) return (a.units > 0) - (b.units > 0);
    if (a.queen != b.queen) return a.queen > b.queen ? 1 : -1;
    if (a.longest != b.longest) return a.longest > b.longest ? 1 : -1;
    if (a.total != b.total) return a.total > b.total ? 1 : -1;
    return 0;
}

int main() {
    Params p;
    const int MAXR = 500;
    std::mt19937 rng(12345);

    // ---- antisymmetry and boundedness, over random shapes, queens and rounds
    float worst_anti = 0.0f, worst_abs = 0.0f;
    for (int trial = 0; trial < 200000; trial++) {
        const int area = 121 + (int)(rng() % 4000);
        const TeamShape us = random_shape(rng, 40, area), them = random_shape(rng, 40, area);
        const int round = (int)(rng() % (MAXR + 1));
        float f[N_TERMS], g[N_TERMS], sf = 0, sg = 0;
        potential(us, them, round, MAXR, area, p, f);
        potential(them, us, round, MAXR, area, p, g);
        for (int i = 0; i < N_TERMS; i++) {
            worst_anti = std::max(worst_anti, std::fabs(f[i] + g[i]));
            sf += f[i];
            sg += g[i];
        }
        worst_abs = std::max(worst_abs, std::fabs(sf));
        check(std::fabs(sf + sg) < 1e-5f, "sum antisymmetry");
    }
    std::printf("antisymmetry: worst |Phi_i(us,them) + Phi_i(them,us)| = %.3e\n", worst_anti);
    std::printf("boundedness:  worst |Phi| = %.4f   (must be <= kappa = %.2f)\n", worst_abs, p.kappa);
    check(worst_anti < 1e-5f, "per-term antisymmetry");
    check(worst_abs <= p.kappa + 1e-5f, "|Phi| <= kappa");

    // ---- identical teams are exactly zero, so a symmetric spawn has Phi = 0
    for (int round : {0, 1, 137, 250, 499, 500}) {
        const TeamShape s = shape({8, 6, 5, 4}, 300);
        check(std::fabs(potential_sum(s, s, round, MAXR, 1024, p)) < 1e-6f, "identical teams give Phi = 0");
    }

    // ---- the end is the verdict: at round 500 only WIN is left, so the sign of Phi is who wins
    {
        int decided = 0, wrong = 0, wrong_queen_gap = 0;
        for (int trial = 0; trial < 200000; trial++) {
            const TeamShape us = random_shape(rng, 60, 1024), them = random_shape(rng, 60, 1024);
            const int v = verdict(us, them);
            if (v == 0 || us.units == 0 || them.units == 0) continue;
            decided++;
            const float phi = potential_sum(us, them, MAXR, MAXR, 1024, p);
            if ((phi > 0) != (v > 0)) {
                wrong++;
                if (us.queen != them.queen) wrong_queen_gap++;
            }
        }
        std::printf("round 500: Phi's sign disagrees with the verdict in %d of %d decided games "
                    "(%.3f%%; %d of them on a queen gap)\n", wrong, decided, 100.0 * wrong / decided,
                    wrong_queen_gap);
        check(wrong == 0, "Phi's sign at the end is exactly the verdict");
    }

    // ---- what losing the queen costs, through the game (queens equal otherwise)
    {
        const int rounds[6] = {0, 100, 250, 375, 450, 500};
        float prev = 0.0f;
        std::printf("losing our queen:");
        for (int r : rounds) {
            const TeamShape us = shape({10, 8}, 4096), foe = shape({10, 8}, 4096);
            const TeamShape us_dead = shape({10, 8}, 4096, -1);
            const float cost = potential_sum(us_dead, foe, r, MAXR, 4096, p) - potential_sum(us, foe, r, MAXR, 4096, p);
            std::printf("  r%d %+.3f", r, cost);
            check(cost < prev - 1e-4f, "a lost queen costs more the later it is");
            prev = cost;
        }
        std::printf("\n");
        check(prev < -0.9f, "at the end a lost queen is nearly a certain loss");
        const Lambdas l375 = lambdas(375, MAXR, 0, 1), l450 = lambdas(450, MAXR, 0, 1), l500 = lambdas(500, MAXR, 0, 1);
        check(std::fabs(l375.v[T_QUEEN] - 1.0f) < 1e-6f && std::fabs(l450.v[T_QUEEN] - 0.4f) < 1e-6f &&
                  l500.v[T_QUEEN] == 0.0f, "QUEEN weight: 1 to round 375, 0 at 500");
        // both queens dead: QUEEN is 0 and WIN falls through to the longest dragon
        float t[N_TERMS];
        raw_terms(shape({12, 8}, 0, -1), shape({10, 8}, 0, -1), p, t);
        check(t[T_QUEEN] == 0.0f && t[T_WIN] > 0.0f, "both queens dead: the longest decides");
    }

    // ---- WIN is monotone in each of the verdict's quantities, the others held,
    // over many configurations (queens >= 2: a living dragon is at least 2 long)
    {
        bool mono = true;
        for (int trial = 0; trial < 20000 && mono; trial++) {
            const int their_q = 2 + (int)(rng() % 40), their_l = their_q + (int)(rng() % 30);
            const int our_l = 2 + (int)(rng() % 70), extra = (int)(rng() % 50);
            float prev_q = -2.0f, prev_l = -2.0f;
            for (int q = 2; q <= our_l; q++) {           // our queen grows inside our longest
                float t[N_TERMS];
                raw_terms(shape({q, our_l, extra + 2}, 0), shape({their_q, their_l, 5}, 0), p, t);
                if (t[T_WIN] < prev_q - 1e-6f) mono = false;
                prev_q = t[T_WIN];
            }
            for (int l = 2; l <= 80; l++) {               // our longest grows, queen fixed
                float t[N_TERMS];
                raw_terms(shape({2, l, extra + 2}, 0), shape({their_q, their_l, 5}, 0), p, t);
                if (t[T_WIN] < prev_l - 1e-6f) mono = false;
                prev_l = t[T_WIN];
            }
        }
        check(mono, "T_WIN monotone in our queen and in our longest");
        float t[N_TERMS];
        raw_terms(shape({20, 40}, 0), shape({21}, 0), p, t);
        check(t[T_WIN] < 0.0f, "a longer dragon does not outvote a longer queen (20 v 21, longest 40 v 21)");
    }

    // ---- the finisher rises monotonically as the enemy's material falls, at
    // every scale, and is maximal at elimination
    for (float c : {5.0f, 15.0f, 40.0f}) {
        Params q = p;
        q.kill_c = c;
        float prev = -2.0f;
        for (int their_total = 100; their_total >= 0; their_total -= 5) {
            float t[N_TERMS];
            raw_terms(shape({20, 10, 5}, 0), their_total > 0 ? shape({their_total}, 0) : shape({}, 0), q, t);
            check(t[T_KILL] > prev, "finisher rises as their material falls");
            prev = t[T_KILL];
        }
        float wipe[N_TERMS], parity[N_TERMS];
        raw_terms(shape({20, 10, 5}, 0), shape({}, 0), q, wipe);
        raw_terms(shape({20, 10, 5}, 0), shape({20, 10, 5}, 0), q, parity);
        check(wipe[T_KILL] > 0.0f, "finisher is positive at a wipe-out");
        check(std::fabs(parity[T_KILL]) < 1e-6f, "finisher is zero at parity");
    }

    // ---- and it is EXACTLY split-invariant (written over total length, not unit count)
    for (float c : {5.0f, 15.0f, 40.0f}) {
        Params q = p;
        q.kill_c = c;
        float before[N_TERMS], after[N_TERMS];
        raw_terms(shape({20, 14, 9, 6, 4}, 0), shape({22, 15}, 0), q, before);
        raw_terms(shape({20, 14, 9, 6, 2, 2}, 0), shape({22, 15}, 0), q, after);
        check(std::fabs(after[T_KILL] - before[T_KILL]) < 1e-6f, "finisher is split-invariant");
    }

    // ---- a free split of a non-leader, non-queen dragon pays nothing (v0's shredding)
    for (int round : {0, 100, 250, 400, 500}) {
        const TeamShape before = shape({20, 14, 9, 6, 4}, 400);
        const TeamShape after = shape({20, 14, 9, 6, 2, 2}, 400);   // split the 4
        const TeamShape foe = shape({22, 15, 10, 5, 3}, 400);
        const float d = potential_sum(after, foe, round, MAXR, 1024, p) - potential_sum(before, foe, round, MAXR, 1024, p);
        check(std::fabs(d) <= 1e-6f, "a free split of a non-leader does not pay");
    }

    // ---- kills and trades (queens untouched: dragon 0 of each list is the queen)
    {
        const TeamShape us = shape({20, 14, 9, 6, 4}, 400);
        const TeamShape foe = shape({22, 15, 10, 5, 3}, 400);
        const float base = potential_sum(us, foe, 250, MAXR, 1024, p);
        const float pearl = potential_sum(shape({20, 14, 9, 6, 5}, 400), foe, 250, MAXR, 1024, p) - base;
        const float small = potential_sum(us, shape({22, 15, 10, 5}, 400), 250, MAXR, 1024, p) - base;
        const float big = potential_sum(us, shape({22, 10, 5, 3}, 400), 250, MAXR, 1024, p) - base;
        const float queen = potential_sum(us, shape({22, 15, 10, 5, 3}, 400, -1), 250, MAXR, 1024, p) - base;
        std::printf("round 250: pearl %+.4f  kill-small %+.4f  kill-their-15 %+.4f  kill-their-queen %+.4f\n",
                    pearl, small, big, queen);
        check(pearl > 0, "a pearl pays");
        check(big > small, "a bigger kill beats a smaller one");
        check(queen > 2 * big, "their queen is worth far more than a dragon its size");
    }

    // ---- coverage: a fresh tile pays while the map is unknown, nothing once known, nothing after round 75
    {
        const TeamShape foe = shape({8, 6}, 100);
        const float open_gain = potential_sum(shape({8, 6}, 105), foe, 20, MAXR, 1024, p) -
                                potential_sum(shape({8, 6}, 100), foe, 20, MAXR, 1024, p);
        const TeamShape foe_done = shape({8, 6}, 1024);
        const float done_gain = potential_sum(shape({8, 6}, 1024), foe_done, 20, MAXR, 1024, p) -
                                potential_sum(shape({8, 6}, 1019), foe_done, 20, MAXR, 1024, p);
        check(open_gain > 0, "coverage pays while the map is unknown");
        check(std::fabs(done_gain) < std::fabs(open_gain) * 0.05f, "coverage stops paying once the map is known");
        const Lambdas e25 = lambdas(25, MAXR, 0, 4096), e50 = lambdas(50, MAXR, 0, 4096), e75 = lambdas(75, MAXR, 0, 4096);
        check(std::fabs(e25.v[T_EXP] - 1.0f) < 1e-6f && std::fabs(e50.v[T_EXP] - 0.5f) < 1e-6f && e75.v[T_EXP] == 0.0f,
              "coverage fades out over rounds 25-75");
    }

    std::printf(failures ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
