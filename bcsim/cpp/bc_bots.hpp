// Scripted opponents for evaluation.
//
// These are yardsticks, not contenders: each one plays a single idea, and the
// ideas are chosen to be as different from one another as possible, so a
// policy's win rate against each says which kind of play it can handle. They
// read the whole board (the learner cannot), which only makes them harder.
//
// None of this is used in training and nothing here changes a rule.
#pragma once

#include "bc_core.hpp"

#include <random>

namespace bc {

enum BotKind {
    BOT_RANDOM = 0,   // any move that does not die on the spot
    BOT_GREEDY,       // nearest pearl, never splits
    BOT_SPLITTER,     // nearest pearl, splits in half whenever it can
    BOT_BLOCKER,      // parks its body in front of the nearest enemy head
    BOT_PORTAL,       // runs for portals and goes through them
    BOT_COWARD,       // open space, away from enemy heads, pearls only when free
    BOT_COUNT
};

inline const char* bot_name(int k) {
    static const char* names[BOT_COUNT] = {"random", "greedy", "splitter", "blocker",
                                           "portal", "coward"};
    return (k >= 0 && k < BOT_COUNT) ? names[k] : "?";
}

struct BotMove {        // what the vector env turns into an Action
    int kind = 2;       // 0 move, 1 split, 2 suicide
    int dir = 0;        // 0 N, 1 E, 2 S, 3 W
    int split_k = 0;
};

class BotBrain {
public:
    BotMove act(const Game& g, int di, int kind, std::mt19937_64& rng) {
        const MapData& m = *g.map;
        const Dragon& d = g.dragons[di];
        const int area = m.area();
        if ((int)dist_.size() < area) {
            dist_.resize(area);
            first_.resize(area);
            seen_.resize(area);
            queue_.resize(area);
        }

        // the moves that survive this turn, and how much room each leaves
        int dest[4], space[4];
        bool ok[4];
        int n_ok = 0;
        for (int dir = 0; dir < 4; dir++) {
            ok[dir] = safe_step(g, di, dir, dest[dir]);
            space[dir] = ok[dir] ? flood(g, dest[dir], d.len + 12) : -1;
            n_ok += ok[dir];
        }
        BotMove mv;
        if (n_ok == 0) {        // boxed in: any move, it dies either way
            mv.kind = 0;
            mv.dir = dir_index(d.facing);
            return mv;
        }
        // a move is roomy when the dragon fits in what it opens up
        const int need = std::min(d.len + 2, d.len + 12);
        bool roomy[4];
        int n_roomy = 0;
        for (int dir = 0; dir < 4; dir++) {
            roomy[dir] = ok[dir] && space[dir] >= need;
            n_roomy += roomy[dir];
        }
        auto best_space = [&]() {
            int best = -1, bd = 0;
            for (int dir = 0; dir < 4; dir++)
                if (ok[dir] && space[dir] > best) { best = space[dir]; bd = dir; }
            return bd;
        };
        auto pick = [&](int want) {    // take `want` if it is roomy, else the roomiest
            if (want >= 0 && (roomy[want] || (n_roomy == 0 && ok[want]))) return want;
            return best_space();
        };

        const int unit_room = g.alive[d.team] < m.unit_limit;
        switch (kind) {
            case BOT_RANDOM: {
                int choices[4], n = 0;
                for (int dir = 0; dir < 4; dir++) if (ok[dir]) choices[n++] = dir;
                mv.kind = 0;
                mv.dir = choices[rng() % n];
                return mv;
            }
            case BOT_SPLITTER:
                if (unit_room && d.len >= 8) {
                    mv.kind = 1;
                    mv.split_k = d.len / 2;
                    return mv;
                }
                [[fallthrough]];
            case BOT_GREEDY: {
                mv.kind = 0;
                mv.dir = pick(bfs_first(g, di, [&](int t) { return g.pearl[t] != 0; }, area));
                return mv;
            }
            case BOT_BLOCKER: {
                // the cells enemy heads are about to enter
                std::fill(seen_.begin(), seen_.begin() + area, 0);
                bool any = false;
                for (const Dragon& o : g.dragons) {
                    if (!o.alive || o.team == d.team) continue;
                    int x = o.head() % m.w, y = o.head() / m.w;
                    for (int s = 0; s < 2; s++) {
                        int nx, ny;
                        if (!tile_after_step(m, x, y, o.facing, nx, ny)) break;
                        seen_[m.idx(nx, ny)] = 1;
                        any = true;
                        x = nx; y = ny;
                    }
                }
                int want = -1;
                if (any) want = bfs_first(g, di, [&](int t) { return seen_[t] != 0; }, 40);
                if (want < 0)
                    want = bfs_first(g, di, [&](int t) { return g.pearl[t] != 0; }, area);
                mv.kind = 0;
                mv.dir = pick(want);
                return mv;
            }
            case BOT_PORTAL: {
                // standing next to a portal: take it
                const int hx = d.head() % m.w, hy = d.head() / m.w;
                for (int dir = 0; dir < 4; dir++) {
                    if (!roomy[dir]) continue;
                    if (edge_kind(m, hx, hy, dir) == EDGE_PORTAL) {
                        mv.kind = 0;
                        mv.dir = dir;
                        return mv;
                    }
                }
                // otherwise head for the nearest tile with a portal on it,
                // though not the one just used, or it would bounce forever
                const int16_t prev = d.len >= 2 ? d.seg(1) : -1;
                int want = bfs_first(g, di, [&](int t) {
                    if (t == prev) return false;
                    const int x = t % m.w, y = t / m.w;
                    for (int dir = 0; dir < 4; dir++)
                        if (edge_kind(m, x, y, dir) == EDGE_PORTAL) return true;
                    return false;
                }, area);
                if (want < 0)
                    want = bfs_first(g, di, [&](int t) { return g.pearl[t] != 0; }, area);
                mv.kind = 0;
                mv.dir = pick(want);
                return mv;
            }
            case BOT_COWARD:
            default: {
                // score: room first, then distance from the nearest enemy head,
                // a pearl only breaks ties
                int best = -1, bd = best_space();
                for (int dir = 0; dir < 4; dir++) {
                    if (!ok[dir]) continue;
                    const int threat = enemy_head_distance(g, d.team, dest[dir]);
                    const int score = std::min(space[dir], need) * 1000 +
                                      std::min(threat, 12) * 10 + (g.pearl[dest[dir]] ? 1 : 0);
                    if (score > best) { best = score; bd = dir; }
                }
                mv.kind = 0;
                mv.dir = bd;
                return mv;
            }
        }
    }

private:
    static uint8_t edge_kind(const MapData& m, int x, int y, int dir) {
        bool vertical; int ex, ey;
        edge_on_side(m, x, y, dir_char4(dir), vertical, ex, ey);
        const int e = m.idx(ex, ey);
        return vertical ? m.v_kind[e] : m.h_kind[e];
    }

    static char dir_char4(int d) { return d == 0 ? 'N' : d == 1 ? 'E' : d == 2 ? 'S' : 'W'; }
    static int dir_index(char c) { return c == 'N' ? 0 : c == 'E' ? 1 : c == 'S' ? 2 : 3; }

    // One step that the engine would not kill us for. Every occupied tile is
    // fatal (a head-on also kills us), the tail included.
    static bool safe_step(const Game& g, int di, int dir, int& dest) {
        const MapData& m = *g.map;
        const Dragon& d = g.dragons[di];
        int nx, ny;
        if (!tile_after_step(m, d.head() % m.w, d.head() / m.w, dir_char4(dir), nx, ny))
            return false;
        dest = m.idx(nx, ny);
        return g.owner[dest] < 0;
    }

    // Free tiles reachable from `start`, stopping once `cap` is reached.
    int flood(const Game& g, int start, int cap) {
        const MapData& m = *g.map;
        std::fill(seen_.begin(), seen_.begin() + m.area(), 0);
        int head = 0, tail = 0;
        queue_[tail++] = start;
        seen_[start] = 1;
        while (head < tail && tail < cap) {
            const int t = queue_[head++];
            for (int dir = 0; dir < 4; dir++) {
                int nx, ny;
                if (!tile_after_step(m, t % m.w, t / m.w, dir_char4(dir), nx, ny)) continue;
                const int n = m.idx(nx, ny);
                if (seen_[n] || g.owner[n] >= 0) continue;
                seen_[n] = 1;
                queue_[tail++] = n;
            }
        }
        return tail;
    }

    // Breadth first from the head over free tiles; returns the first step
    // toward the nearest tile satisfying `goal`, or -1 within `max_dist`.
    template <class Goal>
    int bfs_first(const Game& g, int di, Goal goal, int max_dist) {
        const MapData& m = *g.map;
        const Dragon& d = g.dragons[di];
        const int area = m.area();
        std::fill(dist_.begin(), dist_.begin() + area, -1);
        int head = 0, tail = 0;
        const int h = d.head();
        dist_[h] = 0;
        first_[h] = -1;
        queue_[tail++] = h;
        while (head < tail) {
            const int t = queue_[head++];
            if (dist_[t] >= max_dist) break;
            for (int dir = 0; dir < 4; dir++) {
                int nx, ny;
                if (!tile_after_step(m, t % m.w, t / m.w, dir_char4(dir), nx, ny)) continue;
                const int n = m.idx(nx, ny);
                if (dist_[n] >= 0 || g.owner[n] >= 0) continue;
                dist_[n] = dist_[t] + 1;
                first_[n] = t == h ? dir : first_[t];
                if (goal(n)) return first_[n];
                queue_[tail++] = n;
            }
        }
        return -1;
    }

    static int enemy_head_distance(const Game& g, int team, int tile) {
        const MapData& m = *g.map;
        const int x = tile % m.w, y = tile / m.w;
        int best = 1 << 20;
        for (const Dragon& o : g.dragons) {
            if (!o.alive || o.team == team) continue;
            int dx = std::abs(o.head() % m.w - x), dy = std::abs(o.head() / m.w - y);
            dx = std::min(dx, m.w - dx);
            dy = std::min(dy, m.h - dy);
            best = std::min(best, dx + dy);
        }
        return best;
    }

    std::vector<int> dist_, first_, queue_;
    std::vector<uint8_t> seen_;
};

}  // namespace bc
