// The BC_PORTALREP packet layout: every portal field value round-trips, alongside random values
// of every other field, and no field disturbs another.
//   g++ -O2 -std=c++17 -DBC_SONAR2 -DBC_PORTALREP tests/test_portal_codec.cpp -o /tmp/tpc && /tmp/tpc
#include <cstdio>
#include <random>
#include "../cpp/bc_sonar2.hpp"

using namespace bc;

int main() {
    std::mt19937_64 rng(1);
    long n = 0, bad = 0;
    for (int px = 0; px < 64; px++)
        for (int py = 0; py < 64; py++)
            for (int dir = 0; dir < 4; dir++)
                for (int box = 0; box < 4; box++)
                    for (int pearls = 0; pearls < 4; pearls++) {
                        s2::Packet p;
                        p.hx = (int)(rng() % 64); p.hy = (int)(rng() % 64); p.len = (int)(rng() % 64);
                        p.queen = rng() & 1;
                        p.enemy = rng() & 1;
                        p.ex = (int)(rng() % 64); p.ey = (int)(rng() % 64);
                        p.enemy_queen = p.enemy && (rng() & 1);
                        p.queen_age = p.enemy_queen ? (int)(rng() % 4) : 0;
                        p.came_from = p.enemy_queen ? (int)(rng() % 4) : 0;
                        p.portal = (px + py + dir + box + pearls) % 7 != 0;   // some packets carry none
                        p.px = px; p.py = py; p.pdir = dir; p.pbox = box; p.ppearls = pearls;
                        const int team = (int)(rng() & 1), round = (int)(rng() % 2000);
                        const uint64_t v = s2::encode(p, team, round);
                        s2::Packet q;
                        int sent = -1;
                        bool ok = s2::decode(v, team, round, q, sent) && sent == round;
                        ok = ok && q.hx == p.hx && q.hy == p.hy && q.len == p.len && q.queen == p.queen
                             && q.enemy == p.enemy && q.enemy_queen == p.enemy_queen;
                        if (p.enemy) ok = ok && q.ex == p.ex && q.ey == p.ey;
                        if (p.enemy_queen) ok = ok && q.queen_age == p.queen_age && q.came_from == p.came_from;
                        ok = ok && q.portal == p.portal;
                        if (p.portal)
                            ok = ok && q.px == px && q.py == py && q.pdir == dir && q.pbox == box && q.ppearls == pearls;
                        // every state bit counts: a dragon's own ray must match itself exactly
                        ok = ok && s2::same_sender(v, s2::encode(p, team, round));
                        n++;
                        if (!ok && bad++ < 5) std::printf("bad: px %d py %d dir %d box %d pearls %d\n", px, py, dir, box, pearls);
                    }
    // and any packet with a portal differs from the same packet without one
    s2::Packet a; a.portal = true; a.px = 0; a.py = 0;
    s2::Packet b;
    if (s2::encode(a, 0, 5) == s2::encode(b, 0, 5)) { std::printf("portal flag lost\n"); bad++; }
    std::printf("%ld packets, %ld bad\n", n, bad);
    return bad ? 1 : 0;
}
