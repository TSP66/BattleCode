// C ABI for the reference-comparison path: one game, driven a turn at a time
// with raw protocol text, so it can be run in lockstep with the real engine.
#include "bc_text.hpp"

#include <cstring>

using namespace bc;

namespace {

struct TextHandle {
    MapData map;
    TurnDriver drv;
    std::string scratch;
    int last_death = 0;
};

int copy_out(const std::string& s, char* buf, int cap) {
    const int n = (int)s.size();
    if (buf && cap > 0) {
        const int k = n < cap ? n : cap;
        memcpy(buf, s.data(), k);
    }
    return n;
}

}  // namespace

extern "C" {

void* bct_create(const char* map_text, unsigned int seed, char* err, int errcap) {
    auto* h = new TextHandle();
    std::string message;
    if (!load_map(map_text, h->map, message)) {
        copy_out(message, err, errcap);
        if (err && errcap > 0) err[message.size() < (size_t)errcap ? message.size() : errcap - 1] = 0;
        delete h;
        return nullptr;
    }
    if (err && errcap > 0) err[0] = 0;
    h->drv.start(h->map, seed);
    return h;
}

void bct_destroy(void* p) { delete (TextHandle*)p; }

int bct_next(void* p) { return ((TextHandle*)p)->drv.next_turn(); }

int bct_round(void* p) { return ((TextHandle*)p)->drv.g.round; }

int bct_dragon_id(void* p, int di) { return ((TextHandle*)p)->drv.g.dragons[di].id; }

int bct_num_dragons(void* p) { return (int)((TextHandle*)p)->drv.g.dragons.size(); }

int bct_init_block(void* p, int di, char* buf, int cap) {
    auto* h = (TextHandle*)p;
    return copy_out(render_init_block(h->drv.g, di), buf, cap);
}

int bct_round_block(void* p, int di, char* buf, int cap) {
    auto* h = (TextHandle*)p;
    return copy_out(render_round_block(h->drv.g, di), buf, cap);
}

void bct_reply(void* p, int di, const char* text) {
    auto* h = (TextHandle*)p;
    h->drv.finish_turn(di, parse_reply(text));
}

// Appends (id, round, reason) for deaths since the last call. Returns triples.
int bct_deaths(void* p, int* out, int cap) {
    auto* h = (TextHandle*)p;
    const auto& log = h->drv.g.death_log;
    int n = 0;
    for (size_t i = h->last_death; i + 2 < log.size() + 0 && n < cap; i += 3, n++) {
        out[n * 3 + 0] = log[i];
        out[n * 3 + 1] = log[i + 1];
        out[n * 3 + 2] = log[i + 2];
    }
    h->last_death += n * 3;
    return n;
}

// rounds, winner (-1 draw), end_reason, a_dragons, b_dragons, a_len, b_len
void bct_result(void* p, int* out) {
    auto* h = (TextHandle*)p;
    const Game& g = h->drv.g;
    int count[2] = {0, 0}, total[2] = {0, 0}, longest[2] = {0, 0};
    for (const Dragon& d : g.dragons) {
        if (!d.alive) continue;
        count[d.team]++;
        total[d.team] += d.len;
        longest[d.team] = longest[d.team] > d.len ? longest[d.team] : d.len;
    }
    out[0] = g.round;
    out[1] = g.winner;
    out[2] = g.end_reason;
    out[3] = count[0];
    out[4] = count[1];
    out[5] = total[0];
    out[6] = total[1];
    out[7] = longest[0];
    out[8] = longest[1];
}

}  // extern "C"

extern "C" int bct_stats(void* p, long long* out, int cap) {
    auto* h = (TextHandle*)p;
    const int n = bc::ST_COUNT < cap ? bc::ST_COUNT : cap;
    for (int i = 0; i < n; i++) out[i] = h->drv.g.stats[i];
    return n;
}
