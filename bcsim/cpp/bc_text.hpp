// Map files and the wire protocol, byte compatible with the engine.
#pragma once

#include "bc_core.hpp"

#include <sstream>

namespace bc {

// ------------------------------------------------------------- map files
struct MapError { std::string message; };

inline bool load_map(const std::string& text, MapData& m, std::string& err) {
    m = MapData();
    std::vector<std::array<int, 3>> raw_edges;  // index, kind, portal id
    int tile_count = -1, edge_count = -1, dragon_count = -1;
    bool have_map = false;

    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        std::istringstream ls(line);
        std::string tok;
        if (!(ls >> tok)) continue;

        if (tok == "MAP") {
            if (!(ls >> m.w >> m.h)) { err = "MAP needs width and height"; return false; }
            if (m.w < 1 || m.h < 1 || m.w > 64 || m.h > 64) { err = "map out of range"; return false; }
            have_map = true;
            const int n = m.area();
            m.min_gap.assign(n, 0);
            m.max_gap.assign(n, 0);
            m.spawns.assign(n, 0);
            m.h_kind.assign(n, EDGE_OPEN);
            m.v_kind.assign(n, EDGE_OPEN);
            m.h_portal_id.assign(n, -1);
            m.v_portal_id.assign(n, -1);
            m.h_target.assign(n, PortalTarget());
            m.v_target.assign(n, PortalTarget());
        } else if (!have_map) {
            err = "MAP must come before " + tok;
            return false;
        } else if (tok == "MAP_NAME") {
            std::getline(ls, m.name);
        } else if (tok == "SYMMETRY") {
            std::string s;
            ls >> s;
            // "x" mirrors across the x axis, which flips y, and vice versa.
            if (s == "x") m.sym = SYM_FLIP_Y;
            else if (s == "y") m.sym = SYM_FLIP_X;
            else if (s == "xy") m.sym = SYM_ROT180;
            else { err = "SYMMETRY must be x, y or xy: " + s; return false; }
        } else if (tok == "TILE_COUNT") {
            ls >> tile_count;
        } else if (tok == "EDGE_COUNT") {
            ls >> edge_count;
        } else if (tok == "DRAGON_COUNT") {
            ls >> dragon_count;
        } else if (tok == "UNIT_LIMIT") {
            ls >> m.unit_limit;
        } else if (tok == "TILE") {
            int x, y, mn, mx;
            if (!(ls >> x >> y >> mn >> mx)) { err = "TILE needs x y minGap maxGap"; return false; }
            if (x < 0 || y < 0 || x >= m.w || y >= m.h) { err = "TILE out of bounds"; return false; }
            const int t = m.idx(x, y);
            m.min_gap[t] = mn;
            m.max_gap[t] = mx;
            m.spawns[t] = (mx != 0);
            if (mx != 0 && mx < mn) { err = "TILE maxGap below minGap"; return false; }
        } else if (tok == "EDGE") {
            int index, kind, pid = -1;
            if (!(ls >> index >> kind)) { err = "EDGE needs index kind portalId"; return false; }
            ls >> pid;
            raw_edges.push_back({index, kind, pid});
        } else if (tok == "DRAGON") {
            int team, count;
            if (!(ls >> team >> count)) { err = "DRAGON needs team and segment count"; return false; }
            if (team != 0 && team != 1) { err = "DRAGON team must be 0 or 1"; return false; }
            if (count < 2) { err = "DRAGON is too short"; return false; }
            DragonSpawn s;
            s.team = team;
            for (int i = 0; i < count; i++) {
                int x, y;
                if (!(ls >> x >> y)) { err = "DRAGON ran out of segments"; return false; }
                if (x < 0 || y < 0 || x >= m.w || y >= m.h) { err = "DRAGON segment out of bounds"; return false; }
                s.body.push_back((int16_t)m.idx(x, y));
            }
            m.dragons.push_back(std::move(s));
        }
    }
    if (!have_map) { err = "no MAP directive"; return false; }
    if (tile_count >= 0 || edge_count >= 0 || dragon_count >= 0) { /* counts are advisory here */ }

    // Edge indices: rows alternate horizontal (a tile row's north side) and
    // vertical (its west side); each row is width + 1 long and the last column
    // duplicates the wrap, so it is dropped, as is the final horizontal row.
    struct Pending { bool vertical; int x, y; int pid; };
    std::vector<Pending> portals;
    for (auto& e : raw_edges) {
        const int index = e[0], kind = e[1], pid = e[2];
        const int stride = m.w + 1;
        const int row = index / stride, rem = index - row * stride;
        if (index < 0 || row > 2 * m.h) continue;
        if (rem == m.w) continue;                 // east border, same as west
        const bool vertical = (row & 1) != 0;
        if (!vertical && row == 2 * m.h) continue;  // south border, same as north
        const int x = rem, y = vertical ? (row - 1) / 2 : row / 2;
        const int t = m.idx(x, y);
        uint8_t k = kind == 1 ? EDGE_KELP : (kind == 2 ? EDGE_PORTAL : EDGE_OPEN);
        if (vertical) { m.v_kind[t] = k; m.v_portal_id[t] = (k == EDGE_PORTAL ? pid : -1); }
        else          { m.h_kind[t] = k; m.h_portal_id[t] = (k == EDGE_PORTAL ? pid : -1); }
        if (k == EDGE_PORTAL) {
            if (pid < 0) { err = "portal EDGE needs a portal id"; return false; }
            portals.push_back({vertical, x, y, pid});
        }
    }
    for (size_t i = 0; i < portals.size(); i++) {
        int partner = -1;
        for (size_t j = 0; j < portals.size(); j++)
            if (j != i && portals[j].pid == portals[i].pid) {
                if (partner >= 0) { err = "more than two edges share a portal id"; return false; }
                partner = (int)j;
            }
        if (partner < 0) { err = "portal has no partner"; return false; }
        const Pending& a = portals[i];
        const Pending& b = portals[partner];
        if (a.vertical != b.vertical) { err = "paired portals differ in orientation"; return false; }
        PortalTarget t;
        t.vertical = b.vertical ? 1 : 0;
        t.x = (int16_t)b.x;
        t.y = (int16_t)b.y;
        const int cell = m.idx(a.x, a.y);
        if (a.vertical) m.v_target[cell] = t;
        else            m.h_target[cell] = t;
    }
    return true;
}

// ------------------------------------------------------------- rendering
inline std::string render_init_block(const Game& g, int di) {
    const Dragon& d = g.dragons[di];
    char buf[128];
    int n = snprintf(buf, sizeof buf, "ID %d\nTEAM %c\nMAP %d %d\nUNIT_LIMIT %d\n",
                     d.id, d.team == 0 ? 'A' : 'B', g.map->w, g.map->h, g.map->unit_limit);
    return std::string(buf, n);
}

inline void append_edge_symbol(std::string& out, const MapData& m, bool vertical, int x, int y) {
    const int t = m.idx(x, y);
    const uint8_t k = vertical ? m.v_kind[t] : m.h_kind[t];
    if (k == EDGE_KELP) { out += 'w'; return; }
    if (k == EDGE_PORTAL) {
        out += std::to_string(vertical ? m.v_portal_id[t] : m.h_portal_id[t]);
        return;
    }
    out += '.';
}

// The 7x7 window, the bodies inside it and the edges around it, exactly as
// BuildRoundBlock lays them out.
inline std::string render_round_block(const Game& g, int di) {
    const MapData& m = *g.map;
    const Dragon& d = g.dragons[di];
    const int hx = d.head() % m.w, hy = d.head() / m.w;

    std::string out;
    out.reserve(1024);
    out += "ROUND " + std::to_string(g.round) + "\n";
    out += "DIR "; out += d.facing; out += "\n";
    out += "LENGTH " + std::to_string(d.len) + "\n";
    out += "UNIT_COUNT " + std::to_string(g.alive[d.team]) + "\n";
    out += "NUM_MSGS " + std::to_string(d.inbox.size()) + "\n";
    for (uint32_t v : d.inbox) out += std::to_string(v) + "\n";

    for (int dy = -VISION; dy <= VISION; dy++)
        for (int dx = -VISION; dx <= VISION; dx++) {
            const int x = m.wrapx(hx + dx), y = m.wrapy(hy + dy);
            const int t = m.idx(x, y);
            out += std::to_string(x) + " " + std::to_string(y) + " " +
                   std::to_string((int)g.pearl[t]) + " " + std::to_string(g.cd[t]) + "\n";
        }

    // Bodies: every segment of every living dragon whose cell is in the window,
    // in dragon order, head first.
    std::string bodies;
    int count = 0;
    for (const Dragon& o : g.dragons) {
        if (!o.alive) continue;
        for (int i = 0; i < o.len; i++) {
            const int16_t c = o.seg(i);
            const int x = c % m.w, y = c / m.w;
            int ddx = x - hx, ddy = y - hy;
            if (ddx > m.w / 2) ddx -= m.w;
            if (ddx < -(m.w - 1) / 2) ddx += m.w;
            if (ddy > m.h / 2) ddy -= m.h;
            if (ddy < -(m.h - 1) / 2) ddy += m.h;
            if (ddx < -VISION || ddx > VISION || ddy < -VISION || ddy > VISION) continue;
            char facing = o.facing;
            if (i > 0) {
                const int16_t prev = o.seg(i - 1);
                facing = direction_between(m, x, y, prev % m.w, prev / m.w);
            }
            bodies += (o.team == 0 ? "A " : "B ");
            bodies += std::to_string(o.id) + " " + std::to_string(x) + " " + std::to_string(y) + " ";
            bodies += facing;
            bodies += (i == 0 ? " 1\n" : " 0\n");
            count++;
        }
    }
    out += "DRAGON_BODIES " + std::to_string(count) + "\n" + bodies;

    for (int r = 0; r <= WINDOW; r++) {       // 8 rows of horizontal edges
        for (int c = 0; c < WINDOW; c++) {
            if (c) out += ' ';
            append_edge_symbol(out, m, false, m.wrapx(hx - VISION + c), m.wrapy(hy - VISION + r));
        }
        out += '\n';
    }
    for (int r = 0; r < WINDOW; r++) {        // 7 rows of vertical edges
        for (int c = 0; c <= WINDOW; c++) {
            if (c) out += ' ';
            append_edge_symbol(out, m, true, m.wrapx(hx - VISION + c), m.wrapy(hy - VISION + r));
        }
        out += '\n';
    }
    return out;
}

// ------------------------------------------------------------- replies
enum ActionKind : uint8_t { ACT_SUICIDE = 0, ACT_MOVE = 1, ACT_SPLIT = 2 };

struct Reply {
    uint8_t kind = ACT_SUICIDE;
    std::vector<char> dirs;
    int split = 0;
    bool has_sonar = false;
    uint32_t sonar = 0;
};

inline bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

// sscanf's "%u" / "%d": optional sign, then digits, saturating on overflow the
// way the C library the engine is built against does, and wrapping a negative.
inline bool scan_integer(const char* p, const char*& end, bool is_signed, uint32_t& out) {
    while (is_space(*p)) p++;
    bool neg = false;
    if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
    if (*p < '0' || *p > '9') return false;
    // The C library accumulates in the widest unsigned type, saturating only
    // there, and the conversion then truncates to 32 bits. So 4294967296 is
    // read as 0, not as the largest value.
    uint64_t v = 0;
    bool saturated = false;
    while (*p >= '0' && *p <= '9') {
        const int digit = *p - '0';
        if (v > (0xFFFFFFFFFFFFFFFFull - (uint64_t)digit) / 10ull) saturated = true;
        if (!saturated) v = v * 10ull + (uint64_t)digit;
        p++;
    }
    end = p;
    if (saturated) v = 0xFFFFFFFFFFFFFFFFull;
    (void)is_signed;
    uint32_t truncated = (uint32_t)v;
    out = neg ? (uint32_t)(0u - truncated) : truncated;
    return true;
}

inline bool rest_is_blank(const char* p) {
    while (*p) {
        if (!is_space(*p)) return false;
        p++;
    }
    return true;
}

// Reads lines until ENDTURN. MOVE / SPLIT / SONAR overwrite each other, the
// last one read wins, and a line that will not parse is skipped whole: the
// engine demands that nothing but whitespace follows the argument, so
// "MOVE NN N" is not a two step sprint, it is no action at all.
inline Reply parse_reply(const std::string& text) {
    Reply r;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t nl = text.find('\n', pos);
        std::string line = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;

        const char* p = line.c_str();
        while (is_space(*p)) p++;
        const char* cmd_start = p;
        while (*p && !is_space(*p)) p++;
        std::string cmd(cmd_start, p - cmd_start);
        if (cmd.empty()) continue;
        if (cmd.size() > 15) cmd.resize(15);   // "%15s"

        if (cmd == "ENDTURN") break;
        if (cmd == "MOVE") {
            while (is_space(*p)) p++;
            const char* arg = p;
            while (*p && !is_space(*p)) p++;
            if (arg == p || !rest_is_blank(p)) continue;
            std::vector<char> dirs;
            bool ok = true;
            for (const char* q = arg; q != p; q++) {
                if (*q != 'N' && *q != 'E' && *q != 'S' && *q != 'W') { ok = false; break; }
                dirs.push_back(*q);
            }
            if (!ok) continue;
            r.kind = ACT_MOVE;
            r.dirs = std::move(dirs);
        } else if (cmd == "SPLIT") {
            const char* end = nullptr;
            uint32_t v;
            if (!scan_integer(p, end, true, v) || !rest_is_blank(end)) continue;
            r.kind = ACT_SPLIT;
            r.split = (int32_t)v;
        } else if (cmd == "SONAR") {
            const char* end = nullptr;
            uint32_t v;
            if (!scan_integer(p, end, false, v) || !rest_is_blank(end)) continue;
            r.has_sonar = true;
            r.sonar = v;
        }
        // INDICATOR, LOG, DOT and LINE only affect the replay.
    }
    return r;
}

// ------------------------------------------------------------- turn / round
inline void apply_reply(Game& g, int di, const Reply& r) {
    switch (r.kind) {
        case ACT_MOVE:  g.move(di, r.dirs.data(), (int)r.dirs.size()); break;
        case ACT_SPLIT: g.split(di, r.split); break;
        default:        g.kill(di, DEATH_ACTION); break;
    }
    if (r.has_sonar && g.dragons[di].alive) g.cast_sonar(di, r.sonar);
}

// A game stepped one dragon turn at a time, the way the engine drives bots.
struct TurnDriver {
    Game g;
    int cursor = 0;
    bool round_open = false;

    void start(const MapData& m, uint32_t seed) {
        g.reset(m, seed);
        cursor = 0;
        round_open = false;
    }

    // Finds the next living dragon to act. Returns -1 when the game is over.
    int next_turn() {
        while (!g.finished) {
            if (!round_open) {
                g.pearl_tick();
                round_open = true;
                cursor = 0;
            }
            while (cursor < (int)g.dragons.size()) {
                if (g.dragons[cursor].alive) return cursor;
                cursor++;
            }
            g.settle(false);
            if (g.finished) return -1;
            g.round++;
            round_open = false;
        }
        return -1;
    }

    void finish_turn(int di, const Reply& r) {
        g.dragons[di].inbox.clear();
        apply_reply(g, di, r);
        cursor = di + 1;
    }
};

}  // namespace bc
