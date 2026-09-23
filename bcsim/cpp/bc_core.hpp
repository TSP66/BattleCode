// Exact reimplementation of the UNSW Battlecode engine rules.
//
// Every rule here was taken from the shipped engine (unswbc_engine.wasm,
// engine 0.3.0) by disassembly, and is checked turn by turn against that
// engine by tests/test_oracle.py. Where the two could differ, the engine wins:
// change this file, never the test.
//
// Conventions kept identical to the engine:
//   * directions are the ASCII letters 'N' 'E' 'S' 'W'
//   * a horizontal edge (x, y) is the north side of tile (x, y)
//   * a vertical edge   (x, y) is the west  side of tile (x, y)
//   * pearl countdowns are drawn as  mt19937() % (max - min + 1) + min
//   * tiles are visited row-major, and a tile whose mirror comes earlier is
//     skipped, so the mirror inherits the draw
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace bc {

constexpr int MAX_UNITS_HARD = 4096;   // safety cap, real limit comes from the map
constexpr int VISION = 3;              // 7x7 window
constexpr int WINDOW = 2 * VISION + 1;

// ---------------------------------------------------------------- mt19937
// std::mt19937 as libc++ implements it; the engine seeds it with 1592614637.
struct MT19937 {
    static constexpr uint32_t DEFAULT_SEED = 1592614637u;
    uint32_t mt[624];
    int idx = 625;

    void seed(uint32_t s) {
        mt[0] = s;
        for (uint32_t i = 1; i < 624; i++)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
        idx = 624;
    }
    void generate() {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (mt[i] & 0x80000000u) + (mt[(i + 1) % 624] & 0x7fffffffu);
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1) mt[i] ^= 2567483615u;
        }
        idx = 0;
    }
    uint32_t next() {
        if (idx >= 624) generate();
        uint32_t y = mt[idx++];
        y ^= y >> 11;
        y ^= (y << 7) & 2636928640u;
        y ^= (y << 15) & 4022730752u;
        y ^= y >> 18;
        return y;
    }
};

// ---------------------------------------------------------------- map data
enum EdgeKind : uint8_t { EDGE_OPEN = 0, EDGE_KELP = 1, EDGE_PORTAL = 2 };

// The five counts of the protocol-3 ECHOES line, in its order.
enum SonarEcho : uint8_t {
    SE_KELP = 0, SE_ALLY = 1, SE_ALLY_HEAD = 2, SE_ENEMY = 3, SE_ENEMY_HEAD = 4,
    SONAR_ECHO_KINDS = 5
};
constexpr int SONAR_DIRS = 4;          // one message per cardinal direction a turn
enum Symmetry : uint8_t { SYM_NONE = 0, SYM_FLIP_Y = 1, SYM_FLIP_X = 2, SYM_ROT180 = 3 };

struct PortalTarget {  // partner edge of a portal edge
    int8_t vertical = -1;  // 0 horizontal, 1 vertical, -1 none
    int16_t x = 0, y = 0;
};

struct DragonSpawn {
    int team = 0;               // 0 = A, 1 = B
    std::vector<int16_t> body;  // cell indices, head first
};

// Immutable, shared by every game played on this map.
struct MapData {
    int w = 0, h = 0;
    uint8_t sym = SYM_NONE;
    int unit_limit = 64;
    std::string name;

    std::vector<int32_t> min_gap, max_gap;  // per tile
    std::vector<uint8_t> spawns;            // per tile: draws countdowns at all

    // per tile: the edge on that tile's north / west side
    std::vector<uint8_t> h_kind, v_kind;
    std::vector<int32_t> h_portal_id, v_portal_id;  // -1 when not a portal
    std::vector<PortalTarget> h_target, v_target;

    std::vector<DragonSpawn> dragons;

    int area() const { return w * h; }
    int idx(int x, int y) const { return y * w + x; }
    int wrapx(int x) const { return ((x % w) + w) % w; }
    int wrapy(int y) const { return ((y % h) + h) % h; }

    void mirror(int x, int y, int& mx, int& my) const {
        switch (sym) {
            case SYM_FLIP_Y: mx = x;         my = h - 1 - y; break;
            case SYM_FLIP_X: mx = w - 1 - x; my = y;         break;
            case SYM_ROT180: mx = w - 1 - x; my = h - 1 - y; break;
            default:         mx = x;         my = y;         break;
        }
    }
};

// The edge on the given side of a tile, as the engine's EdgeOnTileSide does.
// Returns vertical=false for a horizontal edge.
inline void edge_on_side(const MapData& m, int x, int y, char dir,
                         bool& vertical, int& ex, int& ey) {
    x = m.wrapx(x); y = m.wrapy(y);
    switch (dir) {
        case 'N': vertical = false; ex = x;                ey = y;                break;
        case 'S': vertical = false; ex = x;                ey = m.wrapy(y + 1);   break;
        case 'W': vertical = true;  ex = x;                ey = y;                break;
        default:  vertical = true;  ex = m.wrapx(x + 1);   ey = y;                break;  // 'E'
    }
}

// Where a step lands, or blocked=false when kelp is in the way. Mirrors
// TileAfterStep: a portal hands you out of its partner edge, keeping heading.
inline bool tile_after_step(const MapData& m, int x, int y, char dir, int& nx, int& ny) {
    bool vertical; int ex, ey;
    edge_on_side(m, x, y, dir, vertical, ex, ey);
    const int e = m.idx(ex, ey);
    const uint8_t kind = vertical ? m.v_kind[e] : m.h_kind[e];
    if (kind == EDGE_KELP) return false;
    if (kind == EDGE_PORTAL) {
        const PortalTarget& t = vertical ? m.v_target[e] : m.h_target[e];
        int tx = t.x, ty = t.y;
        if (t.vertical == 0) ty -= (dir != 'S') ? 1 : 0;
        else                 tx -= (dir != 'E') ? 1 : 0;
        nx = m.wrapx(tx); ny = m.wrapy(ty);
        return true;
    }
    int dx = (dir == 'E') - (dir == 'W');
    int dy = (dir == 'S') - (dir == 'N');
    nx = m.wrapx(x + dx); ny = m.wrapy(y + dy);
    return true;
}

// The direction that steps from a to b, searched N, E, S, W like the engine.
// Returns 0 when no step connects them.
inline char direction_between(const MapData& m, int ax, int ay, int bx, int by) {
    static const char dirs[4] = {'N', 'E', 'S', 'W'};
    for (char d : dirs) {
        int nx, ny;
        if (tile_after_step(m, ax, ay, d, nx, ny) && nx == bx && ny == by) return d;
    }
    return 0;
}

// ---------------------------------------------------------------- dragons
enum DeathReason : uint8_t {
    DEATH_NONE = 0,
    DEATH_WALL = 'W', DEATH_SELF = 'S', DEATH_OTHER = 'O',
    DEATH_HEAD = 'H', DEATH_ACTION = 'A',
};

// Body as a ring buffer of cell indices, segment 0 = head.
struct Dragon {
    int id = 0;
    uint8_t team = 0;
    uint8_t alive = 1;
    char facing = 'N';
    uint8_t death = DEATH_NONE;

    std::vector<int16_t> ring;
    int start = 0, len = 0, mask = 0;
    // Protocol 3 carries the whole word; the legacy form capped it at 32 bits.
    //
    // A message is delivered the moment the ray lands, and every inbox is
    // emptied at the *round* boundary -- not when the dragon reads it. So a
    // dragon hears what was cast earlier in the same round, before its own
    // turn, and never hears what was cast after it. Measured both ways against
    // the engine (tests/parity_sonar.py): clearing per turn delivers a round
    // late, and holding everything to the boundary delivers a round early.
    std::vector<uint64_t> inbox;
    // What this dragon's own sonars hit, in the order of the ECHOES line:
    // kelp, ally, ally_head, enemy, enemy_head. Each ray lands in exactly one,
    // so these sum to the number of sonars sent -- measured against the engine,
    // see SONAR.md. Filled while the dragon acts and reported in its next block.
    int32_t echo[SONAR_ECHO_KINDS] = {};
    // The protocol this dragon has declared. It is per dragon, not per team:
    // the engine runs one bot instance per dragon (it spawns them by dragon id),
    // so a dragon born from a split starts on the legacy protocol until its own
    // first reply declares otherwise.
    uint8_t protocol = 2;

    void reserve_ring(int want) {
        int cap = 8;
        while (cap < want) cap <<= 1;
        if ((int)ring.size() >= cap) return;
        std::vector<int16_t> next(cap);
        for (int i = 0; i < len; i++) next[i] = ring[(start + i) & mask];
        ring.swap(next);
        start = 0;
        mask = cap - 1;
    }
    int16_t seg(int i) const { return ring[(start + i) & mask]; }
    int16_t head() const { return ring[start & mask]; }
    int16_t tail() const { return ring[(start + len - 1) & mask]; }
    void push_front(int16_t cell) {
        if (len + 1 > (int)ring.size()) reserve_ring(len + 1);
        start = (start - 1) & mask;
        ring[start] = cell;
        len++;
    }
    void pop_back() { len--; }
};

// --------------------------------------------------------------- events
enum EventKind : uint8_t {
    EV_DEATH = 0, EV_SPLIT = 1, EV_PEARL_EATEN = 2, EV_STEP = 3, EV_SONAR_HIT = 4,
};
struct Event {
    uint8_t kind;
    int32_t a, b, c, d;  // death(id, reason, len, killer id), split(parent, child, k, -)
};

// Counters proving which rule paths a test actually exercised.
enum Stat {
    ST_STEP = 0, ST_PORTAL_STEP, ST_SPRINT_STEP, ST_PEARL_EATEN, ST_PEARL_SPAWN,
    ST_DEATH_WALL, ST_DEATH_SELF, ST_DEATH_OTHER, ST_DEATH_HEAD, ST_DEATH_ACTION,
    ST_SPLIT_OK, ST_SPLIT_ILLEGAL, ST_SPLIT_LIMIT, ST_SONAR_CAST, ST_SONAR_HIT,
    ST_SONAR_SELF, ST_SONAR_LOST, ST_BLOCKED_SPAWN, ST_COUNT
};

// --------------------------------------------------------------- game
struct Game {
    const MapData* map = nullptr;
    MT19937 rng;

    std::vector<uint8_t> pearl;    // per tile, 0/1
    std::vector<int32_t> cd;       // per tile countdown, -1 = never
    std::vector<int16_t> owner;    // per tile, dragon index or -1
    std::vector<uint8_t> head_at;  // per tile, 1 when the occupant's head is here
    std::vector<uint8_t> seg_dir;  // per tile, the way that segment points (0 N, 1 E, 2 S, 3 W)

    std::vector<Dragon> dragons;
    int alive[2] = {0, 0};
    int round = 0;
    bool finished = false;
    int winner = -1;     // -1 draw / none, 0 = A, 1 = B
    int end_reason = 0;  // 0 elimination, 1 length

    // per-turn bookkeeping the vector env turns into rewards
    std::vector<Event> events;
    std::vector<int32_t> death_log;  // id, round, reason triples
    int64_t stats[ST_COUNT] = {0};
    bool record_events = true;

    void reset(const MapData& m, uint32_t seed) {
        map = &m;
        rng.seed(seed);
        const int n = m.area();
        pearl.assign(n, 0);
        cd.assign(n, -1);
        owner.assign(n, -1);
        head_at.assign(n, 0);
        seg_dir.assign(n, 0);
        dragons.clear();
        dragons.reserve(64);
        alive[0] = alive[1] = 0;
        round = 0;
        finished = false;
        winner = -1;
        end_reason = 0;
        events.clear();
        death_log.clear();
        for (int i = 0; i < ST_COUNT; i++) stats[i] = 0;

        for (const DragonSpawn& s : m.dragons) {
            Dragon d;
            d.id = (int)dragons.size();
            d.team = (uint8_t)s.team;
            d.reserve_ring((int)s.body.size());
            for (size_t i = 0; i < s.body.size(); i++) {
                d.ring[i] = s.body[i];
                d.len++;
            }
            d.start = 0;
            // The head's facing comes from the step that leads into it.
            if (d.len >= 2) {
                int hx = s.body[0] % m.w, hy = s.body[0] / m.w;
                int nx = s.body[1] % m.w, ny = s.body[1] / m.w;
                char back = direction_between(m, hx, hy, nx, ny);
                d.facing = opposite(back);
            }
            dragons.push_back(std::move(d));
            Dragon& nd = dragons.back();
            alive[nd.team]++;
            for (int i = 0; i < nd.len; i++) {
                owner[nd.seg(i)] = (int16_t)(dragons.size() - 1);
                head_at[nd.seg(i)] = (i == 0);
                char face = nd.facing;
                if (i > 0) {
                    const int16_t a = nd.seg(i), b = nd.seg(i - 1);
                    face = direction_between(m, a % m.w, a / m.w, b % m.w, b / m.w);
                }
                seg_dir[nd.seg(i)] = dir_code(face);
            }
        }
        init_pearl_countdowns();
    }

    static uint8_t dir_code(char d) {
        return d == 'N' ? 0 : d == 'E' ? 1 : d == 'S' ? 2 : 3;
    }

    static char opposite(char d) {
        switch (d) {
            case 'N': return 'S';
            case 'S': return 'N';
            case 'E': return 'W';
            default:  return 'E';
        }
    }

    // ---- pearls
    void init_pearl_countdowns() {
        const MapData& m = *map;
        for (int y = 0; y < m.h; y++)
            for (int x = 0; x < m.w; x++) {
                int mx, my;
                m.mirror(x, y, mx, my);
                if (m.idx(mx, my) < m.idx(x, y)) continue;
                const int t = m.idx(x, y);
                if (!m.spawns[t]) continue;
                int32_t v = (int32_t)(rng.next() % (uint32_t)(m.max_gap[t] - m.min_gap[t] + 1))
                            + m.min_gap[t];
                cd[t] = v;
                cd[m.idx(mx, my)] = v;
            }
    }

    void pearl_tick() {
        const MapData& m = *map;
        for (int y = 0; y < m.h; y++)
            for (int x = 0; x < m.w; x++) {
                int mx, my;
                m.mirror(x, y, mx, my);
                const int t = m.idx(x, y), tm = m.idx(mx, my);
                if (tm < t) continue;
                if (!m.spawns[t]) continue;
                const int32_t old = cd[t];
                cd[t] = old - 1;
                cd[tm] = old - 1;
                if (old > 1) continue;
                if (!pearl[t] && owner[t] < 0) { pearl[t] = 1; stats[ST_PEARL_SPAWN]++; }
                else stats[ST_BLOCKED_SPAWN]++;
                if (tm != t) {
                    if (!pearl[tm] && owner[tm] < 0) { pearl[tm] = 1; stats[ST_PEARL_SPAWN]++; }
                    else stats[ST_BLOCKED_SPAWN]++;
                }
                int32_t v = (int32_t)(rng.next() % (uint32_t)(m.max_gap[t] - m.min_gap[t] + 1))
                            + m.min_gap[t];
                cd[t] = v;
                cd[tm] = v;
            }
    }

    // ---- deaths
    void kill(int di, uint8_t reason, int killer = -1) {
        Dragon& d = dragons[di];
        if (!d.alive) return;
        if (record_events)
            events.push_back({EV_DEATH, d.id, reason, d.len,
                              killer >= 0 ? dragons[killer].id : -1});
        switch (reason) {
            case DEATH_WALL:   stats[ST_DEATH_WALL]++; break;
            case DEATH_SELF:   stats[ST_DEATH_SELF]++; break;
            case DEATH_OTHER:  stats[ST_DEATH_OTHER]++; break;
            case DEATH_HEAD:   stats[ST_DEATH_HEAD]++; break;
            case DEATH_ACTION: stats[ST_DEATH_ACTION]++; break;
            default: break;
        }
        death_log.push_back(d.id);
        death_log.push_back(round);
        death_log.push_back(reason);
        for (int i = 0; i < d.len; i++) {
            const int16_t c = d.seg(i);
            owner[c] = -1;
            head_at[c] = 0;
        }
        for (int i = 0; i < d.len; i += 2) pearl[d.seg(i)] = 1;
        d.alive = 0;
        d.death = reason;
        alive[d.team]--;
    }

    // ---- movement
    // One step. `extra` is true for every step after the first in a sprint,
    // which costs one more tail segment. Returns false once the dragon died.
    bool step(int di, char dir, bool extra) {
        Dragon& d = dragons[di];
        const MapData& m = *map;
        d.facing = dir;
        const int16_t from = d.head();
        int nx, ny;
        {
            bool vertical; int ex, ey;
            edge_on_side(m, from % m.w, from / m.w, dir, vertical, ex, ey);
            const int e = m.idx(ex, ey);
            if ((vertical ? m.v_kind[e] : m.h_kind[e]) == EDGE_PORTAL) stats[ST_PORTAL_STEP]++;
        }
        if (!tile_after_step(m, from % m.w, from / m.w, dir, nx, ny)) {
            kill(di, DEATH_WALL);
            return false;
        }
        const int16_t dest = (int16_t)m.idx(nx, ny);

        const int16_t occ = owner[dest];
        if (occ == (int16_t)di) {          // own body, tail included
            kill(di, DEATH_SELF);
            return false;
        }
        if (occ >= 0) {
            if (head_at[dest]) {           // the other dragon dies first
                kill(occ, DEATH_HEAD, di);
                kill(di, DEATH_HEAD, occ);
            } else {
                kill(di, DEATH_OTHER, occ);
            }
            return false;
        }

        d.push_front(dest);
        owner[dest] = (int16_t)di;
        head_at[dest] = 1;
        seg_dir[dest] = dir_code(dir);
        if (d.len >= 2) {
            head_at[d.seg(1)] = 0;
            seg_dir[d.seg(1)] = dir_code(dir);   // the old head now points the way it went
        }

        bool ate = false;
        if (pearl[dest]) {
            pearl[dest] = 0;
            ate = true;
            if (record_events) events.push_back({EV_PEARL_EATEN, d.id, dest, 0, 0});
        } else {
            drop_tail(d);
        }
        if (extra) { drop_tail(d); stats[ST_SPRINT_STEP]++; }
        stats[ST_STEP]++;
        if (ate) stats[ST_PEARL_EATEN]++;
        if (record_events) events.push_back({EV_STEP, d.id, dest, ate, 0});
        return true;
    }

    void drop_tail(Dragon& d) {
        const int16_t t = d.tail();
        d.pop_back();
        // Only clear the grid if no earlier segment still sits there. A body
        // cannot self-overlap, so the tail's cell is free once it leaves.
        owner[t] = -1;
        head_at[t] = 0;
    }

    // MOVE with one or more steps. An empty list is not an action at all.
    void move(int di, const char* dirs, int n) {
        if (n <= 0) return;
        if (!step(di, dirs[0], false)) return;
        for (int i = 1; i < n; i++) {
            if (dragons[di].len <= 2) {   // cannot pay for another step
                kill(di, DEATH_ACTION);
                return;
            }
            if (!step(di, dirs[i], true)) return;
        }
    }

    // SPLIT k: the rear k segments leave as a new dragon, reversed.
    void split(int di, int k) {
        Dragon& parent = dragons[di];
        const MapData& m = *map;
        const int len = parent.len;
        if (k < 2 || k > len - 2 || alive[parent.team] >= m.unit_limit) {
            stats[alive[parent.team] >= m.unit_limit ? ST_SPLIT_LIMIT : ST_SPLIT_ILLEGAL]++;
            kill(di, DEATH_ACTION);
            return;
        }
        stats[ST_SPLIT_OK]++;
        Dragon child;
        child.id = (int)dragons.size();
        child.team = parent.team;
        child.reserve_ring(k);
        for (int i = 0; i < k; i++) child.ring[i] = parent.seg(len - 1 - i);
        child.start = 0;
        child.len = k;
        {
            const int16_t a = child.ring[0], b = child.ring[1];
            char back = direction_between(m, a % m.w, a / m.w, b % m.w, b / m.w);
            child.facing = opposite(back);
        }
        for (int i = 0; i < k; i++) parent.pop_back();

        const int ci = (int)dragons.size();
        dragons.push_back(std::move(child));
        Dragon& c = dragons[ci];
        for (int i = 0; i < c.len; i++) {
            owner[c.seg(i)] = (int16_t)ci;
            head_at[c.seg(i)] = (i == 0);
            char face = c.facing;
            if (i > 0) {
                const int16_t a = c.seg(i), b = c.seg(i - 1);
                face = direction_between(m, a % m.w, a / m.w, b % m.w, b / m.w);
            }
            seg_dir[c.seg(i)] = dir_code(face);
        }
        alive[c.team]++;
        if (record_events) events.push_back({EV_SPLIT, dragons[di].id, c.id, k, 0});
    }

    // SONAR: a ray from the head along `facing`, through portals and around the
    // wrap, stopping at kelp or the first living dragon.
    //
    // Under protocol 3 the direction is chosen rather than taken from the
    // dragon's facing, the payload is a full 64 bits, and the ray reports back:
    // whatever it stopped on is counted into the sender's `echo`. Measured
    // against the reference engine (SONAR.md): each ray lands in exactly one of
    // the five categories, so the counts sum to the number of sonars sent, and
    // a kelp edge stops the ray before a dragon on the far side of it.
    //
    // Whoever the ray stops on receives the message -- ally, enemy or, when the
    // ray wraps the torus, the sender itself. There is no privacy here.
    void cast_sonar(int di, char facing, uint64_t value) {
        const MapData& m = *map;
        Dragon& d = dragons[di];
        const int limit = m.w + m.h;
        stats[ST_SONAR_CAST]++;
        int x = d.head() % m.w, y = d.head() / m.w;
        // The ray steps out through the sender's own body without seeing it, and
        // once clear of it the body becomes an ordinary target again -- which
        // matters on a torus, where a ray with nothing in its way comes back
        // round and hits the dragon that sent it. Measured both ways against the
        // engine: stopping at the body immediately gets the backward ray wrong,
        // never stopping at it gets the forward ray wrong (which then finds
        // nothing at all on an open map), and this gets both.
        bool clear_of_self = false;
        for (int i = 1; i <= limit; i++) {
            int nx, ny;
            if (!tile_after_step(m, x, y, facing, nx, ny)) {
                stats[ST_SONAR_LOST]++;
                d.echo[SE_KELP]++;
                return;
            }
            x = nx; y = ny;
            const int16_t occ = owner[m.idx(x, y)];
            // Under protocol 3 a dragon's own body is transparent to its own
            // sonar: the ray goes straight through and neither stops there nor
            // delivers. Measured against the engine -- a dragon whose southward
            // ray runs down its own tail is told "kelp", where counting the tail
            // would have said "ally" (tests/parity_sonar.py).
            //
            // The legacy protocol does *not* do this: the ray stops on the
            // sender's own body and the sender receives its own message, which
            // is what tests/test_vecenv.py checks against the engine. So this is
            // one of the things 1.0.0 changed, not a rule we had wrong before.
            if (occ >= 0 && occ == di && d.protocol >= 3 && !clear_of_self) continue;
            if (occ != di) clear_of_self = true;
            if (occ >= 0) {
                if (occ == di) stats[ST_SONAR_SELF]++;
                stats[ST_SONAR_HIT]++;
                const bool head = head_at[m.idx(x, y)] != 0;
                const bool ally = dragons[occ].team == d.team;
                d.echo[ally ? (head ? SE_ALLY_HEAD : SE_ALLY)
                            : (head ? SE_ENEMY_HEAD : SE_ENEMY)]++;
                dragons[occ].inbox.push_back(value);
                if (record_events)
                    events.push_back({EV_SONAR_HIT, d.id, dragons[occ].id,
                                      (int32_t)(uint32_t)value, 0});
                return;
            }
        }
        // Nothing in w + h steps. On a torus a straight ray comes back to its
        // own body long before this, so it takes a short dragon on an empty
        // line to get here; the engine's behaviour in this case is unmeasured,
        // and no category is counted.
        stats[ST_SONAR_LOST]++;
    }

    // The legacy protocol-2 form: along the dragon's own facing, 32 bits.
    void cast_sonar(int di, uint32_t value) {
        cast_sonar(di, dragons[di].facing, (uint64_t)value);
    }



    // ---- outcome, exactly ResultAfterRound
    void settle(bool force_end) {
        int count[2] = {0, 0}, longest[2] = {0, 0}, total[2] = {0, 0};
        for (const Dragon& d : dragons) {
            if (!d.alive) continue;
            count[d.team]++;
            total[d.team] += d.len;
            longest[d.team] = std::max(longest[d.team], d.len);
        }
        if (count[0] == 0 || count[1] == 0) {
            finished = true;
            end_reason = 0;
            winner = (count[0] == count[1]) ? -1 : (count[0] ? 0 : 1);
            return;
        }
        if (!force_end && round < 499) return;
        finished = true;
        end_reason = 1;
        if (longest[0] != longest[1]) winner = longest[0] > longest[1] ? 0 : 1;
        else if (total[0] != total[1]) winner = total[0] > total[1] ? 0 : 1;
        else winner = -1;
    }

    int alive_units(int team) const { return alive[team]; }
};

}  // namespace bc
