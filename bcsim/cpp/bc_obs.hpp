// Observations and action masks for the batched environment.
//
// Two rules govern everything here:
//   1. the acting dragon may only be shown what the protocol would tell it,
//      so a policy trained on this can be deployed unchanged;
//   2. the action mask is computed the same way a deployed bot would compute
//      it, from the window plus the dragon's own body, so masked training does
//      not become a crutch that disappears at submission time.
#pragma once

#include "bc_vec.hpp"

namespace bc {

// Turns an offset in the dragon's own frame into a board offset.
inline void ego_to_world(int facing, int ox, int oy, int& wx, int& wy) {
    switch (facing) {
        case 0: wx = ox;  wy = oy;  break;   // N: no rotation
        case 1: wx = -oy; wy = ox;  break;   // E
        case 2: wx = -ox; wy = -oy; break;   // S
        default: wx = oy; wy = -ox; break;   // W
    }
}

// ... and back again.
inline void world_to_ego(int facing, int wx, int wy, int& ox, int& oy) {
    switch (facing) {
        case 0: ox = wx;  oy = wy;  break;
        case 1: ox = wy;  oy = -wx; break;
        case 2: ox = -wx; oy = -wy; break;
        default: ox = -wy; oy = wx; break;
    }
}

// The critic view (layout: bc_vec.hpp, cview_stride) cut from a rendered board.
inline void VecEnv::write_cview(const uint8_t* bd, int hx, int hy, int w, int h, uint8_t* out) const {
    const size_t plane = (size_t)BOARD_MAX * BOARD_MAX;
    const int W = cview_w_, r = W / 2;
    memset(out, 0, cview_stride(W));
    uint8_t* bytes = out + cview_bits_bytes(W);
    for (int iy = 0; iy < W; iy++) {
        const int y = ((hy - r + iy) % h + h) % h;
        for (int ix = 0; ix < W; ix++) {
            const int x = ((hx - r + ix) % w + w) % w;
            const size_t at = (size_t)y * BOARD_MAX + x;
            for (int p = 0; p < CV_BITS; p++) {
                if (bd[CV_SRC[p] * plane + at]) {
                    const size_t k = ((size_t)p * W + iy) * W + ix;
                    out[k >> 3] |= (uint8_t)(1u << (k & 7));
                }
            }
            for (int p = CV_BITS; p < CV_CH; p++)
                bytes[((size_t)(p - CV_BITS) * W + iy) * W + ix] = bd[CV_SRC[p] * plane + at];
        }
    }
    // Only blocks that touch the map can be non-zero (the board is zero outside it, and `out`
    // was cleared). Binary planes hold exactly 0/1 (test_cview.py checks against (v != 0) * 255).
    uint8_t* coarse = out + cview_coarse_off(W);
    const int nby = (h + 3) / 4, nbx = (w + 3) / 4;
    for (int p = 0; p < CV_CH; p++) {
        const uint8_t* src = bd + CV_SRC[p] * plane;
        const unsigned scale = p < CV_BITS ? 255u : 1u;
        for (int by = 0; by < nby; by++) {
            unsigned acc[CV_COARSE] = {0};
            for (int dy = 0; dy < 4; dy++) {
                const uint8_t* row = src + (size_t)(by * 4 + dy) * BOARD_MAX;
                for (int bx = 0; bx < nbx; bx++)
                    acc[bx] += (unsigned)row[bx * 4] + row[bx * 4 + 1] + row[bx * 4 + 2] + row[bx * 4 + 3];
            }
            uint8_t* dst = coarse + ((size_t)p * CV_COARSE + by) * CV_COARSE;
            for (int bx = 0; bx < nbx; bx++) dst[bx] = (uint8_t)((acc[bx] * scale + 8) / 16);
        }
    }
}

inline void VecEnv::observe(Env& e, int index) {
    const Game& g = e.game;
    const MapData& m = *e.map;
    const int di = e.acting;
    const Dragon& d = g.dragons[di];
    const int facing = cfg_.egocentric ? dir_index(d.facing) : 0;
    const int hx = d.head() % m.w, hy = d.head() / m.w;

    // what this dragon remembers, updated from the same window it is shown
    DragonMemory& dm = e.mem_for(d.id, m.w, m.h);

    float* local = b_local_ + (size_t)index * LC_COUNT * WINDOW * WINDOW;
    memset(local, 0, sizeof(float) * LC_COUNT * WINDOW * WINDOW);
    auto at = [&](int channel, int row, int col) -> float& {
        return local[(size_t)channel * WINDOW * WINDOW + row * WINDOW + col];
    };

    // own body, so index and tail can be marked without peeking at anything
    // the dragon does not already know about itself
    for (int i = 0; i < d.len; i++) {
        const int16_t c = d.seg(i);
        int ddx = c % m.w - hx, ddy = c / m.w - hy;
        if (ddx > m.w / 2) ddx -= m.w;
        if (ddx < -(m.w - 1) / 2) ddx += m.w;
        if (ddy > m.h / 2) ddy -= m.h;
        if (ddy < -(m.h - 1) / 2) ddy += m.h;
        if (ddx < -VISION || ddx > VISION || ddy < -VISION || ddy > VISION) continue;
        int ox, oy;
        world_to_ego(facing, ddx, ddy, ox, oy);
        const int row = oy + VISION, col = ox + VISION;
        at(LC_SELF_INDEX, row, col) = (float)i / (float)std::max(1, d.len - 1);
        if (i == d.len - 1) at(LC_SELF_TAIL, row, col) = 1.0f;
    }

    // the queens' segments in the window (0 ours, 1 theirs): the protocol names every
    // visible segment's dragon id, and ids 0 and 1 are the queens
    float qwin[2][WINDOW * WINDOW] = {};
    if (Game::is_queen(d)) dm.saw_queen(0, hx, hy, g.round, true);   // a queen knows where it is

    for (int row = 0; row < WINDOW; row++)
        for (int col = 0; col < WINDOW; col++) {
            int wx, wy;
            ego_to_world(facing, col - VISION, row - VISION, wx, wy);
            const int x = m.wrapx(hx + wx), y = m.wrapy(hy + wy);
            const int t = m.idx(x, y);

            at(LC_PEARL, row, col) = (float)g.pearl[t];
            if (g.cd[t] < 0) at(LC_NEVER_SPAWN, row, col) = 1.0f;
            else at(LC_PEARL_TIME, row, col) = std::min(g.cd[t], 99) / 99.0f;

            const int16_t occ = g.owner[t];
            if (occ >= 0) {
                const Dragon& o = g.dragons[occ];
                const bool head = g.head_at[t] != 0;
                if (occ == di) at(head ? LC_SELF_HEAD : LC_SELF_BODY, row, col) = 1.0f;
                else if (o.team == d.team) {
                    at(head ? LC_ALLY_HEAD : LC_ALLY_BODY, row, col) = 1.0f;
                    dm.saw_ally(x, y, g.round);  // only the grid reads this
                }
                else {
                    at(head ? LC_ENEMY_HEAD : LC_ENEMY_BODY, row, col) = 1.0f;
                    dm.saw_foe(x, y, g.round);   // only the wide planes read this
                }
                const int world_dir = g.seg_dir[t];
                const int shown = (world_dir - facing + 4) % 4;
                at(LC_FACE_N + shown, row, col) = 1.0f;
                if (occ != di && Game::is_queen(o)) {
                    const int side = o.team == d.team ? 0 : 1;
                    qwin[side][row * WINDOW + col] = 1.0f;
                    dm.saw_queen(side, x, y, g.round, head, s2::bearing(wx, wy));
                }
            }

            bool kelp_here = false;
            uint8_t sides = 0;                    // world directions, for the grid
            for (int o_dir = 0; o_dir < 4; o_dir++) {
                const int world_dir = (o_dir + facing) % 4;
                bool vertical; int ex, ey;
                edge_on_side(m, x, y, dir_char(world_dir), vertical, ex, ey);
                const int edge = m.idx(ex, ey);
                const uint8_t kind = vertical ? m.v_kind[edge] : m.h_kind[edge];
                if (kind == EDGE_KELP) {
                    at(LC_KELP_N + o_dir, row, col) = 1.0f;
                    kelp_here = true;
                    sides |= (uint8_t)(1 << world_dir);
                } else if (kind == EDGE_PORTAL) {
                    at(LC_PORTAL_N + o_dir, row, col) = 1.0f;
                    sides |= (uint8_t)(1 << (4 + world_dir));
                }
            }
            dm.see_sides(x, y, sides);
            // MemoryTracker reads these three off the planes; taking them from
            // the same game values avoids a float round-trip and is identical
            dm.see(x, y, g.round, g.pearl[t] != 0, g.cd[t], kelp_here);
        }

    float* sc = b_scalar_ + (size_t)index * SC_TOTAL;
    memset(sc, 0, sizeof(float) * SC_TOTAL);
    sc[SC_ROUND] = (float)g.round / (float)cfg_.max_rounds;
    sc[SC_LENGTH] = std::min(d.len, 64) / 64.0f;
    sc[SC_LENGTH_RAW] = (float)d.len;
    sc[SC_UNITS] = (float)g.alive[d.team] / (float)m.unit_limit;
    sc[SC_FACE_N + dir_index(d.facing)] = 1.0f;
    sc[SC_HEAD_X] = (float)hx / (float)m.w;
    sc[SC_HEAD_Y] = (float)hy / (float)m.h;
    sc[SC_MAP_W] = (float)m.w / 64.0f;
    sc[SC_MAP_H] = (float)m.h / 64.0f;
    // Saturates at NUM_MSGS_CAP (4), not at MAX_MSGS: mybot/obs.hpp saturates
    // its own copy of this feature at 4, and widening the message buffer must
    // not move a column every existing checkpoint was trained to read. The true
    // count is reported separately, through bind_num_msgs.
    sc[SC_NUM_MSGS] = (float)std::min<size_t>(d.inbox.size(), NUM_MSGS_CAP);
    sc[SC_TEAM_B] = d.team == 1 ? 1.0f : 0.0f;

    // the head's cell, then the remembered features -- in MemoryTracker's
    // order: every window cell, then where the head stands, then read back.
    // The ego frame uses the dragon's true facing, which is what the tracker
    // takes from the one-hot face scalars, not this file's `facing` (which is
    // zero when the window is not egocentric).
    dm.stand(hx, hy, g.round);
    dm.features(hx, hy, dir_index(d.facing), g.round, sc + SC_COUNT);

    // The same memory as planes, for a net with a conv branch over it. Bound
    // only when something asks, so the 708-scalar checkpoints pay nothing.
    if (b_wide_)
        dm.wide(hx, hy, dir_index(d.facing), g.round,
                b_wide_ + (size_t)index * wide_cfg::N_WIDE);

#ifdef BC_SONAR2
    // Sonar v2 (bc_sonar2.hpp): this turn's verified team packets. Their heads and
    // enemy sightings go into this dragon's memory, so the grid's ally/enemy memory
    // planes show them wherever they are; the rest is drawn below at each head.
    s2::Packet rep[MAX_MSGS];
    int n_rep = 0;
    if ((e.s2_mask >> d.team) & 1) {
        for (size_t i = 0; i < d.inbox.size() && n_rep < MAX_MSGS; i++) {
            s2::Packet p;
            int sent = -1;
            if (!s2::decode(d.inbox[i], d.team, g.round, p, sent)) continue;
            if (p.hx >= m.w || p.hy >= m.h) continue;
            if ((size_t)di < e.s2_last.size() && s2::same_sender(d.inbox[i], e.s2_last[(size_t)di])) continue;   // its own ray
            bool dup = false;
            for (int j = 0; j < n_rep; j++) dup |= rep[j].hx == p.hx && rep[j].hy == p.hy;
            if (dup) continue;                                   // four rays, one sender
            dm.heard_ally(p.hx, p.hy, sent);
            if (p.enemy && p.ex < m.w && p.ey < m.h) dm.heard_foe(p.ex, p.ey, sent);
            if (p.queen) dm.heard_queen(0, p.hx, p.hy, sent);
            if (p.enemy_queen && p.ex < m.w && p.ey < m.h) {
                // the offset from the sender to this dragon, on the torus
                int sx = hx - p.hx, sy = hy - p.hy;
                if (sx > m.w / 2) sx -= m.w;
                if (sx < -(m.w - 1) / 2) sx += m.w;
                if (sy > m.h / 2) sy -= m.h;
                if (sy < -(m.h - 1) / 2) sy += m.h;
                // a relayed report that came from this dragon's side is its own news coming back
                const bool echo = p.queen_age > 0 && s2::bearing(sx, sy) == p.came_from;
                if (!echo) dm.heard_queen(1, p.ex, p.ey, sent - p.queen_age, s2::bearing(-sx, -sy));
            }
#ifdef BC_PORTALREP
            // a teammate's last portal: kept unless this dragon has seen that entrance and it
            // has no portal on that side (then the packet is a forgery or a tag collision)
            if (p.portal && p.px < m.w && p.py < m.h && !dm.knows_no_portal(p.px, p.py, p.pdir))
                dm.heard_portal(p.px, p.py, p.pdir, p.pbox, p.ppearls, sent);
#endif
            rep[n_rep++] = p;
        }
    }
#endif
#ifdef BC_PORTALREP
    // its own last portal, as it would report it (bc_vec.hpp s2_packet makes the same call).
    // Only for a team that speaks the packet, so a silent team's portal planes stay empty
    if ((e.s2_mask >> d.team) & 1) {
        int tiles = 0, pearls = 0;
        if (dm.trip_far_side(g.round, s2::PORTAL_BOX_MAX, s2::PORTAL_PEARL_STEPS, tiles, pearls))
            dm.heard_portal(dm.trip.ax, dm.trip.ay, dm.trip.dir, s2::portal_box_bucket(tiles),
                            std::min(pearls, 3), g.round);
    }
#endif

    // The LSTM policy's 38 x 14 x 14 grid (bc_memory.hpp grid_cfg). Bound only
    // when something asks, so every other architecture pays nothing.
    if (b_grid_) {
        using namespace grid_cfg;
        float* gr = b_grid_ + (size_t)index * grid_cfg::N;
        memset(gr, 0, sizeof(float) * grid_cfg::N);
        dm.grid(hx, hy, facing, g.round, gr);
        auto gch = [&](int c, int r, int col) -> float& {
            return gr[(size_t)c * grid_cfg::CELLS + r * grid_cfg::G + col];
        };
        // live: the window sits at rows/cols HALF-VISION .. HALF+VISION
        static constexpr int LIVE[10][2] = {
            {LC_PEARL, PEARL}, {LC_PEARL_TIME, PEARL_TIMER},
            {LC_ALLY_HEAD, ALLY_HEAD}, {LC_ALLY_BODY, ALLY_BODY},
            {LC_ENEMY_HEAD, ENEMY_HEAD}, {LC_ENEMY_BODY, ENEMY_BODY},
            {LC_FACE_N, SEG_DIR}, {LC_FACE_E, SEG_DIR + 1}, {LC_FACE_S, SEG_DIR + 2}, {LC_FACE_W, SEG_DIR + 3}};
        for (int row = 0; row < WINDOW; row++)
            for (int col = 0; col < WINDOW; col++)
                for (auto const& lv : LIVE)
                    gch(lv[1], row + HALF - VISION, col + HALF - VISION) = at(lv[0], row, col);
        // self: every segment the grid covers, not just the window's. The bot
        // knows its own body from its own moves.
        for (int i = 1; i < d.len; i++) {
            const int16_t c = d.seg(i);
            int ddx = c % m.w - hx, ddy = c / m.w - hy;
            if (ddx > m.w / 2) ddx -= m.w;
            if (ddx < -(m.w - 1) / 2) ddx += m.w;
            if (ddy > m.h / 2) ddy -= m.h;
            if (ddy < -(m.h - 1) / 2) ddy += m.h;
            int ox, oy;
            world_to_ego(facing, ddx, ddy, ox, oy);
            if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
            gch(SELF_BODY, oy + HALF, ox + HALF) = 1.0f;
            gch(SELF_INDEX, oy + HALF, ox + HALF) = (float)i / (float)std::max(1, d.len - 1);
            if (i == d.len - 1) gch(SELF_TAIL, oy + HALF, ox + HALF) = 1.0f;
        }
        const float glob[10] = {
            (float)g.round / (float)cfg_.max_rounds, std::min(d.len, 64) / 64.0f,
            (float)g.alive[d.team] / (float)m.unit_limit, (float)m.w / 64.0f, (float)m.h / 64.0f,
            (float)d.echo[0] / SONAR_DIRS, (float)d.echo[1] / SONAR_DIRS, (float)d.echo[2] / SONAR_DIRS,
            (float)d.echo[3] / SONAR_DIRS, (float)d.echo[4] / SONAR_DIRS};
        for (int k = 0; k < 10; k++) {
            float* p = gr + (size_t)(ROUND + k) * grid_cfg::CELLS;
            for (int i = 0; i < grid_cfg::CELLS; i++) p[i] = glob[k];
        }
#ifdef BC_SONAR2
        // each reporter at its head: length, P(split), P(sprint), and the expected
        // direction of its first step, rotated into this dragon's frame
        static const int UX[4] = {0, 1, 0, -1}, UY[4] = {-1, 0, 1, 0};
        for (int j = 0; j < n_rep; j++) {
            const s2::Packet& p = rep[j];
            int ddx = p.hx - hx, ddy = p.hy - hy;
            if (ddx > m.w / 2) ddx -= m.w;
            if (ddx < -(m.w - 1) / 2) ddx += m.w;
            if (ddy > m.h / 2) ddy -= m.h;
            if (ddy < -(m.h - 1) / 2) ddy += m.h;
            int ox, oy;
            world_to_ego(facing, ddx, ddy, ox, oy);
            if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
            const int r = oy + HALF, c = ox + HALF;
            const int F = p.facing, L = (F + 3) & 3, R = (F + 1) & 3;
            const float ps = std::max(0.0f, 1.0f - p.p_split - p.p_left - p.p_right);
            const float wx = ps * UX[F] + p.p_left * UX[L] + p.p_right * UX[R];
            const float wy = ps * UY[F] + p.p_left * UY[L] + p.p_right * UY[R];
            // world_to_ego on a float vector (the same rotation)
            float ex, ey;
            switch (facing) {
                case 0: ex = wx;  ey = wy;  break;
                case 1: ex = wy;  ey = -wx; break;
                case 2: ex = -wx; ey = -wy; break;
                default: ex = -wy; ey = wx; break;
            }
            gch(REP_LEN, r, c) = std::min(p.len, 63) / 63.0f;
#ifndef BC_PORTALREP
            gch(REP_SPLIT, r, c) = p.p_split;
            gch(REP_SPRINT, r, c) = p.p_sprint;
            gch(REP_DX, r, c) = ex;
            gch(REP_DY, r, c) = ey;
#else
            (void)ex; (void)ey;
#endif
        }
#ifdef BC_PORTALREP
        // what each known portal leads to, on the tile across its edge (grid_cfg PT_*); where
        // two land on one tile, the larger value of each channel
        for (int i = 0; i < dm.n_portal; i++) {
            const DragonMemory::PortalReport& pr = dm.portal_rep[(size_t)i];
            const int bx = pr.tile % m.w + UX[pr.dir], by = pr.tile / m.w + UY[pr.dir];
            int ddx = bx - hx, ddy = by - hy;
            ddx = ((ddx % m.w) + m.w) % m.w;
            ddy = ((ddy % m.h) + m.h) % m.h;
            if (ddx > m.w / 2) ddx -= m.w;
            if (ddy > m.h / 2) ddy -= m.h;
            int ox, oy;
            world_to_ego(facing, ddx, ddy, ox, oy);
            if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
            const int r = oy + HALF, c = ox + HALF;
            const float room = pr.box > 0 ? (float)pr.box / 3.0f : 0.0f;
            const float pearls = (float)pr.pearls / 3.0f * GridDecay::at(GRID_DECAY.pearl, g.round - pr.round);
            gch(PT_KNOWN, r, c) = 1.0f;
            gch(PT_CLOSED, r, c) = std::max(gch(PT_CLOSED, r, c), pr.box > 0 ? 1.0f : 0.0f);
            gch(PT_ROOM, r, c) = std::max(gch(PT_ROOM, r, c), room);
            gch(PT_PEARLS, r, c) = std::max(gch(PT_PEARLS, r, c), pearls);
        }
#endif

        // the queens (grid_cfg IS_QUEEN..EQ_FRESH)
        auto fill = [&](int c, float v) {
            float* pl = gr + (size_t)c * grid_cfg::CELLS;
            for (int i = 0; i < grid_cfg::CELLS; i++) pl[i] = v;
        };
        if (Game::is_queen(d)) fill(IS_QUEEN, 1.0f);
        for (int row = 0; row < WINDOW; row++)
            for (int col = 0; col < WINDOW; col++) {
                gch(ALLY_QUEEN, row + HALF - VISION, col + HALF - VISION) = qwin[0][row * WINDOW + col];
                gch(ENEMY_QUEEN, row + HALF - VISION, col + HALF - VISION) = qwin[1][row * WINDOW + col];
            }
        static constexpr int Q_MEM[2] = {ALLY_QUEEN_MEM, ENEMY_QUEEN_MEM};
        static constexpr int Q_VEC[2] = {AQ_DX, EQ_DX};
        for (int side = 0; side < 2; side++) {
            const DragonMemory::Sighting& q = dm.queen[side];
            if (q.round <= mem_cfg::NEVER) continue;            // never known: all zero
            const float fresh = GridDecay::at(GRID_DECAY.seen, g.round - q.round);
            if (fresh <= 0.0f) continue;
            int ddx = q.x - hx, ddy = q.y - hy;
            if (ddx > m.w / 2) ddx -= m.w;
            if (ddx < -(m.w - 1) / 2) ddx += m.w;
            if (ddy > m.h / 2) ddy -= m.h;
            if (ddy < -(m.h - 1) / 2) ddy += m.h;
            int ox, oy;
            world_to_ego(facing, ddx, ddy, ox, oy);
            fill(Q_VEC[side], std::max(-1.0f, std::min(1.0f, ox / 32.0f)));
            fill(Q_VEC[side] + 1, std::max(-1.0f, std::min(1.0f, oy / 32.0f)));
            fill(Q_VEC[side] + 2, fresh);
            if (ox >= -HALF && ox < G - HALF && oy >= -HALF && oy < G - HALF)
                gch(Q_MEM[side], oy + HALF, ox + HALF) = fresh;
        }
        // who this dragon is (grid_cfg BIRTH..ID43), so it can take on a role; d.id is the engine's id
        fill(BIRTH, (float)d.birth / 500.0f);
        fill(ID7, std::sin((float)d.id / 7.0f));
        fill(ID43, std::sin((float)d.id / 43.0f));
#ifdef BC_AHIST
        static_assert(CODEC_ACTIONS <= DragonMemory::AHIST_MAX && CODEC_ACTIONS <= CELLS, "AHIST plane too small");
        for (int k = 0; k < CODEC_ACTIONS; k++) gch(AHIST, k / G, k % G) = dm.ahist[(size_t)k];
#endif
#endif
    }

    // What this dragon's own sonars came back with last turn. Zero throughout
    // unless the env was built with sonar on, because nothing is cast then.
    for (int k = 0; k < SONAR_ECHO_KINDS; k++)
        sc[SC_ECHO_AT + k] = (float)d.echo[k] / (float)SONAR_DIRS;

    // The payloads themselves, at full width. These were uint32 until the
    // parent-to-child work: the inbox has always been uint64 and the engine
    // delivers 64 bits to a protocol-3 dragon, so every payload above 2^32 was
    // being cut here, at the last step before python could see it.
    uint64_t* msgs = b_msgs_ + (size_t)index * MAX_MSGS;
    for (int i = 0; i < MAX_MSGS; i++)
        msgs[i] = i < (int)d.inbox.size() ? d.inbox[i] : 0ull;
    if (b_nmsgs_) b_nmsgs_[index] = (int32_t)d.inbox.size();

    if (b_priv_) {
        // what decides the game, which the 7x7 window cannot show: both
        // teams' size, their longest dragon and how many units each has
        int tl, tm, tu, fl, fm, fu;
        team_stats(e, d.team, tl, tm, tu);
        team_stats(e, (uint8_t)(1 - d.team), fl, fm, fu);
        float* pv = b_priv_ + (size_t)index * PRIV_COUNT;
        pv[0] = std::log1p((float)tl) / 5.0f;
        pv[1] = std::log1p((float)tm) / 5.0f;
        pv[2] = (float)tu / (float)m.unit_limit;
        pv[3] = std::log1p((float)fl) / 5.0f;
        pv[4] = std::log1p((float)fm) / 5.0f;
        pv[5] = (float)fu / (float)m.unit_limit;
        pv[6] = (float)g.round / (float)cfg_.max_rounds;
        pv[7] = std::tanh((float)(tm - fm) / 10.0f);
        // the queens (2026-10-01): each team's queen length, 0 once it is dead
        const bc8::TeamShape qs = team_shape(e, d.team), qf = team_shape(e, (uint8_t)(1 - d.team));
        pv[8] = std::log1p((float)qs.queen) / 5.0f;
        pv[9] = std::log1p((float)qf.queen) / 5.0f;
        // reward v8: Phi for this dragon's team, per component, as the reward
        // banks it. Zero when v8 is off, so a critic trained without it sees a
        // constant and is unaffected.
        float phi[bc8::N_TERMS] = {0};
        if (cfg_.reward_v8) {
            bc8::potential(team_shape(e, d.team), team_shape(e, (uint8_t)(1 - d.team)),
                           g.round, cfg_.max_rounds, m.area(), cfg_.v8, phi);
        }
        for (int k = 0; k < bc8::N_TERMS; k++) pv[PRIV_BASE + k] = phi[k];
    }

    if (b_board_ || b_cview_) {
        const size_t plane = (size_t)BOARD_MAX * BOARD_MAX;
        // the critic view alone is cut from this thread's own scratch board, so the board
        // has one renderer and the view cannot drift from it
        static thread_local std::vector<uint8_t> scratch;
        uint8_t* bd;
        // the view alone needs CV_SRC's planes: the portal partners (14-17) are drawn for the board only
        const bool full = b_board_ != nullptr;
        if (full) {
            bd = b_board_ + (size_t)index * BOARD_CH * plane;
        } else {
            scratch.resize((size_t)BOARD_CH * plane);
            bd = scratch.data();
        }
        memset(bd, 0, BOARD_CH * plane);
        for (int y = 0; y < m.h && y < BOARD_MAX; y++)
            for (int x = 0; x < m.w && x < BOARD_MAX; x++) {
                const int t = m.idx(x, y);
                const size_t at = (size_t)y * BOARD_MAX + x;
                const int16_t occ = g.owner[t];
                if (occ >= 0) {
                    const bool own = g.dragons[occ].team == d.team;
                    const bool head = g.head_at[t] != 0;
                    bd[(own ? (head ? 1 : 0) : (head ? 3 : 2)) * plane + at] = 1;
                    // Which of our dragons is acting. Planes 0-3 mark every
                    // dragon of each side, so without these the critic sees
                    // the position but not whose turn it is, and one board
                    // would have to serve every dragon on it.
                    if (occ == di) bd[(head ? 9 : 8) * plane + at] = 1;
                    if (Game::is_queen(g.dragons[occ])) bd[(own ? 18 : 19) * plane + at] = 1;
                }
                bd[4 * plane + at] = g.pearl[t];
                bd[5 * plane + at] = 1;
                bd[6 * plane + at] = m.h_kind[t] == EDGE_KELP;
                bd[7 * plane + at] = m.v_kind[t] == EDGE_KELP;
                // portals, laid out like the kelp planes (2026-09-29): without them
                // the critic saw a portal only inside the acting dragon's window
                bd[10 * plane + at] = m.h_kind[t] == EDGE_PORTAL;
                bd[11 * plane + at] = m.v_kind[t] == EDGE_PORTAL;
                // When a pearl is due here, as nearness rather than delay:
                // 255 is due now, small is far off, 0 is never. A countdown
                // past 255 rounds is beyond any game, so it reads as never.
                bd[12 * plane + at] = g.cd[t] < 0 ? 0
                    : (uint8_t)(255 - std::min(g.cd[t], 255));
                // where a portal leads (2026-09-29): the partner edge, always of the same
                // orientation (bc_text rejects mixed pairs), as its tile's x and y scaled
                // 1..255 across the map; 0 = no portal on that edge
                auto enc = [](int v, int n) -> uint8_t {
                    return (uint8_t)(1 + std::lround(254.0 * v / std::max(n - 1, 1)));
                };
                if (full && m.h_kind[t] == EDGE_PORTAL) {
                    bd[14 * plane + at] = enc(m.h_target[t].x, m.w);
                    bd[15 * plane + at] = enc(m.h_target[t].y, m.h);
                }
                if (full && m.v_kind[t] == EDGE_PORTAL) {
                    bd[16 * plane + at] = enc(m.v_target[t].x, m.w);
                    bd[17 * plane + at] = enc(m.v_target[t].y, m.h);
                }
            }
        // Who still moves this round (2026-09-29, user): dragons act in index order,
        // so every living dragon after the acting one -- either team, split children
        // included -- moves before the position is next seen from this side. Each is
        // painted over its whole body at 0.5^(k/32), k = moves before it (the next
        // mover is 255), so the far end of a long queue fades out.
        int k = 0;
        for (size_t j = (size_t)di + 1; j < g.dragons.size(); j++) {
            const Dragon& o = g.dragons[j];
            if (!o.alive) continue;
            const uint8_t v = (uint8_t)std::lround(255.0 * std::exp2(-(double)k / 32.0));
            for (int i = 0; i < o.len; i++) {
                const int c = o.seg(i);
                const int x = c % m.w, y = c / m.w;
                if (x < BOARD_MAX && y < BOARD_MAX)
                    bd[13 * plane + (size_t)y * BOARD_MAX + x] = v;
            }
            k++;
        }
        if (b_cview_) write_cview(bd, hx, hy, std::min(m.w, BOARD_MAX), std::min(m.h, BOARD_MAX),
                                  b_cview_ + (size_t)index * cview_stride(cview_w_));
    }

    b_uid_[index] = (int64_t)e.agents[di].uid;
    b_dragon_[index] = d.id;
    b_team_[index] = (int8_t)d.team;
    b_round_[index] = g.round;

    fill_mask(e, index);
}

// Legality as far as the dragon can tell: kelp and visible bodies block, a
// portal hides what is beyond it so the step counts as allowed, and sprint
// steps have to be paid for.
inline bool VecEnv::path_legal(const Game& g, int di, const char* dirs, int n, bool heads_block) {
    const MapData& m = *g.map;
    const Dragon& d = g.dragons[(size_t)di];
    int x = d.head() % m.w, y = d.head() / m.w;
    const int free = Game::free_steps(d.len);
    int length = d.len, dropped = 0, n_added = 0;
    int16_t added[MAX_STEPS];
    for (int s = 0; s < n; s++) {
        if (s >= free && length <= 2) return false;
        const char dir = dirs[s];
        bool vertical; int ex, ey;
        edge_on_side(m, x, y, dir, vertical, ex, ey);
        const int edge = m.idx(ex, ey);
        const uint8_t kind = vertical ? m.v_kind[edge] : m.h_kind[edge];
        if (kind == EDGE_KELP) return false;
        if (kind == EDGE_PORTAL) {                // cannot see the far side ...
            if (heads_block)                      // ... but a guarded queen's paid steps must still be
                for (int r = s; r < n; r++) {     // affordable, counting no pearls over there
                    if (r >= free && length <= 2) return false;
                    if (r >= free) length--;
                }
            return true;
        }
        int nx = 0, ny = 0;
        tile_after_step(m, x, y, dir, nx, ny);
        const int t = m.idx(nx, ny);

        bool blocked = false;
        for (int j = 0; j < n_added; j++) if (added[j] == (int16_t)t) blocked = true;
        if (!blocked && g.owner[t] == (int16_t)di) {
            blocked = true;                   // still our own body ...
            for (int j = 0; j < dropped; j++)  // ... unless that tail has gone
                if (d.seg(d.len - 1 - j) == (int16_t)t) blocked = false;
        }
        if (!blocked && g.owner[t] >= 0 && g.owner[t] != (int16_t)di && (heads_block || !g.head_at[t]))
            blocked = true;                   // another dragon's body is certain death (its head too, guarded)
        if (blocked) return false;

        if (n_added < MAX_STEPS) added[n_added++] = (int16_t)t;
        if (g.pearl[t]) length++;
        else dropped++;
        if (s >= free) { dropped++; length--; }
        x = nx; y = ny;
    }
    return true;
}

// The far sprints (FAR_TARGETS): ONE breadth-first search over the 7x7 window in the dragon's
// own frame, from its head, stepping forward, left, right, back in that order (so the first
// shortest path in that order wins), gives every target's path. A step must cross an open edge
// (kelp blocks; a portal leads out of the window's geometry, so it blocks too) into a free tile;
// another dragon's head may end a path (a head-on) but is never passed through. Bodies block as
// they stand, tails included. n_out[k] is target k's path length (<= FAR_MAX_STEPS), 0 when it
// has none; its steps go to dirs_out[k * FAR_MAX_STEPS ...]. A search per target, stopping at
// its target, assigns exactly the same parents (tests/test_farsprint.py checks the paths against
// an independent per-target search).
inline void VecEnv::far_paths(const Game& g, int di, int* n_out, char* dirs_out) {
    const MapData& m = *g.map;
    const Dragon& d = g.dragons[(size_t)di];
    const int hx = d.head() % m.w, hy = d.head() / m.w;
    const int facing = dir_index(d.facing);
    static const int EGO_DIR[4] = {0, 3, 1, 2};       // forward (ego N), left (W), right (E), back (S)
    static const int EDX[4] = {0, 1, 0, -1}, EDY[4] = {-1, 0, 1, 0};
    constexpr int W = WINDOW, NC = W * W;
    int8_t from[NC], how[NC], depth[NC];
    for (int i = 0; i < NC; i++) from[i] = -1;
    auto cell = [&](int ox, int oy) { return (oy + VISION) * W + (ox + VISION); };
    auto world = [&](int ox, int oy, int& x, int& y) {
        int wx, wy;
        ego_to_world(facing, ox, oy, wx, wy);
        x = m.wrapx(hx + wx); y = m.wrapy(hy + wy);
    };
    int queue[NC], qh = 0, qt = 0;
    const int start = cell(0, 0);
    from[start] = (int8_t)start;
    depth[start] = 0;
    queue[qt++] = start;
    while (qh < qt) {
        const int c = queue[qh++];
        const int ox = c % W - VISION, oy = c / W - VISION;
        if (depth[c] >= FAR_MAX_STEPS) continue;
        int x, y;
        world(ox, oy, x, y);
        for (int j = 0; j < 4; j++) {
            const int ed = EGO_DIR[j];
            const int nox = ox + EDX[ed], noy = oy + EDY[ed];
            if (nox < -VISION || nox > VISION || noy < -VISION || noy > VISION) continue;
            const int nc = cell(nox, noy);
            if (from[nc] >= 0) continue;
            const char wd = dir_char((ed + facing) & 3);
            bool vertical; int ex, ey;
            edge_on_side(m, x, y, wd, vertical, ex, ey);
            const int edge = m.idx(ex, ey);
            if ((vertical ? m.v_kind[edge] : m.h_kind[edge]) != EDGE_OPEN) continue;
            int nx, ny;
            world(nox, noy, nx, ny);
            const int t = m.idx(nx, ny);
            const int16_t occ = g.owner[t];
            if (occ >= 0 && (occ == (int16_t)di || !g.head_at[t])) continue;   // a body
            from[nc] = (int8_t)c;
            how[nc] = (int8_t)((ed + facing) & 3);
            depth[nc] = (int8_t)(depth[c] + 1);
            if (occ < 0) queue[qt++] = nc;                 // a head ends a path, never passed
        }
    }
    for (int k = 0; k < CODEC_FAR; k++) {
        const int goal = cell(FAR_TARGETS.ox[k], FAR_TARGETS.oy[k]);
        n_out[k] = 0;
        if (from[goal] < 0) continue;
        const int n = depth[goal];
        char* dirs = dirs_out + (size_t)k * FAR_MAX_STEPS;
        int i = n;
        for (int c = goal; c != start; c = from[c]) dirs[--i] = dir_char(how[c]);
        n_out[k] = n;
    }
}

inline int VecEnv::far_path(const Game& g, int di, int k, char* dirs) {
    if (k < 0 || k >= CODEC_FAR) return 0;
    int n[CODEC_FAR > 0 ? CODEC_FAR : 1];
    char all[(CODEC_FAR > 0 ? CODEC_FAR : 1) * FAR_MAX_STEPS];
    far_paths(g, di, n, all);
    for (int s = 0; s < n[k]; s++) dirs[s] = all[(size_t)k * FAR_MAX_STEPS + s];
    return n[k];
}

inline Action VecEnv::decode_full(const Game& g, int di, int action_id) {
    if (!codec_is_far(action_id)) return decode_for(g.dragons[(size_t)di], action_id);
    Action a = decode_for(g.dragons[(size_t)di], -1);      // the self-kill: what a pathless one is
    char dirs[MAX_STEPS];
    const int n = far_path(g, di, action_id - CODEC_FAR_ID, dirs);
    if (n > 0) {
        a.kind = 0;
        a.n_steps = (int8_t)n;
        for (int s = 0; s < n; s++) a.dirs[s] = (int8_t)dir_index(dirs[s]);
    }
    return a;
}

inline void VecEnv::fill_mask(Env& e, int index) {
    const Game& g = e.game;
    const MapData& m = *e.map;
    const int di = e.acting;
    const Dragon& d = g.dragons[di];
    uint8_t* mask = b_mask_ + (size_t)index * CODEC_ACTIONS;
    memset(mask, 0, CODEC_ACTIONS);
    if (CODEC_SUICIDE) mask[CODEC_MOVES + CODEC_SPLITS] = 1;     // always available

    for (int id = 0; id < CODEC_MOVES; id++) {
        const Action a = decode_for(d, id);
        char dirs[MAX_STEPS];
        for (int s = 0; s < a.n_steps; s++) dirs[s] = dir_char(a.dirs[s]);
        mask[id] = path_legal(g, di, dirs, a.n_steps) ? 1 : 0;
    }
    if (CODEC_FAR > 0) {
        int n[CODEC_FAR > 0 ? CODEC_FAR : 1];
        char all[(CODEC_FAR > 0 ? CODEC_FAR : 1) * FAR_MAX_STEPS];
        far_paths(g, di, n, all);
        for (int k = 0; k < CODEC_FAR; k++)
            mask[CODEC_FAR_ID + k] = (n[k] > 0 && path_legal(g, di, all + (size_t)k * FAR_MAX_STEPS, n[k])) ? 1 : 0;
    }

    const bool room = g.alive[d.team] < m.unit_limit;
    for (int i = 0; i < CODEC_SPLITS; i++) {
        const int k = CODEC_SPLIT_K[i] < 0 ? d.len / 2 : CODEC_SPLIT_K[i];
        mask[CODEC_MOVES + i] = (room && k >= 2 && k <= d.len - 2) ? 1 : 0;
    }
    for (int i = 0; i < CODEC_XSPLITS; i++) {
        const int k = d.len - CODEC_XSPLIT_KEEP[i];
        mask[CODEC_XSPLIT_ID + i] = (room && k >= 2 && k <= d.len - 2) ? 1 : 0;
    }

    // the queen guard (VecConfig::queen_guard): no self-kill, no move onto a head -- unless that
    // leaves her nothing, when she keeps the ordinary mask (a forced head-on still takes one with her)
    if (cfg_.queen_guard && Game::is_queen(d)) {
        uint8_t guarded[CODEC_ACTIONS];
        memcpy(guarded, mask, CODEC_ACTIONS);
        if (CODEC_SUICIDE) guarded[CODEC_SUICIDE_ID] = 0;
        for (int id = 0; id < CODEC_MOVES; id++) {
            if (!guarded[id]) continue;
            const Action a = decode_for(d, id);
            char dirs[MAX_STEPS];
            for (int s = 0; s < a.n_steps; s++) dirs[s] = dir_char(a.dirs[s]);
            if (!path_legal(g, di, dirs, a.n_steps, true)) guarded[id] = 0;
        }
        if (CODEC_FAR > 0) {
            int n[CODEC_FAR > 0 ? CODEC_FAR : 1];
            char all[(CODEC_FAR > 0 ? CODEC_FAR : 1) * FAR_MAX_STEPS];
            far_paths(g, di, n, all);
            for (int k = 0; k < CODEC_FAR; k++)
                if (guarded[CODEC_FAR_ID + k] && !path_legal(g, di, all + (size_t)k * FAR_MAX_STEPS, n[k], true))
                    guarded[CODEC_FAR_ID + k] = 0;
        }
        bool any = false;
        for (int id = 0; id < CODEC_ACTIONS; id++) any = any || guarded[id];
        if (any) memcpy(mask, guarded, CODEC_ACTIONS);      // else nothing safe: the ordinary mask
    }
    if (cfg_.queen_deadend && Game::is_queen(d)) queen_deadend_mask(g, di, mask, cfg_.queen_deadend);
}

// Whether queen di in gg stands in a dead end seen from (hx, hy), where she decides: walking on from
// her head, exactly one way at every tile until there is none, every tile within VISION of (hx, hy),
// no portal, no dragon beside the tube (it might move). perfect()'s PP_QUEEN_DEADEND test.
// own_walls (queen dead-end level 2): her own segment s beside the tube tile at walk depth k is a wall
// when it is still there on her k+1-th move from here, len - s > k + 1 (her tail leaves one tile a
// move at most; a pearl only keeps it longer), and within VISION of (hx, hy) (the bot sees no further). Without it any tube she has just walked around reads
// as "might move" (match 1532070, round 143).
inline bool VecEnv::queen_in_dead_end(const Game& gg, int di, int hx, int hy, bool own_walls) {
    const MapData& m = *gg.map;
    const Dragon& q = gg.dragons[(size_t)di];
    if (!q.alive) return false;
    static const char DIRS[4] = {'N', 'E', 'S', 'W'};
    int x = q.head() % m.w, y = q.head() / m.w;
    char came = q.facing;
    for (int depth = 0; depth <= 2 * WINDOW; depth++) {
        int ox = x - hx, oy = y - hy;                    // the obs code's offset on the torus
        if (ox > m.w / 2) ox -= m.w;
        if (ox < -(m.w - 1) / 2) ox += m.w;
        if (oy > m.h / 2) oy -= m.h;
        if (oy < -(m.h - 1) / 2) oy += m.h;
        if (std::abs(ox) > VISION || std::abs(oy) > VISION) return false;
        const char back = Game::opposite(came);
        int ways = 0, nx = 0, ny = 0;
        char nd = 0;
        for (char dir : DIRS) {
            if (dir == back) continue;
            bool vertical; int ex, ey;
            edge_on_side(m, x, y, dir, vertical, ex, ey);
            const uint8_t kind = vertical ? m.v_kind[m.idx(ex, ey)] : m.h_kind[m.idx(ex, ey)];
            if (kind == EDGE_KELP) continue;
            if (kind == EDGE_PORTAL) return false;
            int tx, ty;
            tile_after_step(m, x, y, dir, tx, ty);
            const int16_t occ = gg.owner[m.idx(tx, ty)];
            if (occ >= 0) {
                if (!own_walls || occ != di) return false;
                int wx = tx - hx, wy = ty - hy;                // a wall only where she can see it (the bot)
                if (wx > m.w / 2) wx -= m.w;
                if (wx < -(m.w - 1) / 2) wx += m.w;
                if (wy > m.h / 2) wy -= m.h;
                if (wy < -(m.h - 1) / 2) wy += m.h;
                if (std::abs(wx) > VISION || std::abs(wy) > VISION) return false;
                const int16_t at = (int16_t)m.idx(tx, ty);
                int s = 0;
                while (s < q.len && q.seg(s) != at) s++;
                if (q.len - s > depth + 1) continue;           // still her body when she would get there
                return false;
            }
            ways++;
            nx = tx; ny = ty; nd = dir;
        }
        if (ways == 0) return true;
        if (ways > 1) return false;
        x = nx; y = ny; came = nd;
    }
    return false;
}

// One move of queen di judged on a copy: 1 fatal (dead, or in a dead end she can see from (hx, hy)),
// 0 not, and 0 for a path across a portal (not judged: the bot cannot see where it lets out).
inline int VecEnv::queen_move_fatal(const Game& g, int di, const char* dirs, int n, int hx, int hy,
                                    bool own_walls) {
    const MapData& m = *g.map;
    const Dragon& d = g.dragons[(size_t)di];
    int x = d.head() % m.w, y = d.head() / m.w;
    for (int s = 0; s < n; s++) {
        bool vertical; int ex, ey;
        edge_on_side(m, x, y, dirs[s], vertical, ex, ey);
        if ((vertical ? m.v_kind[m.idx(ex, ey)] : m.h_kind[m.idx(ex, ey)]) == EDGE_PORTAL) return 0;
        int tx, ty;
        tile_after_step(m, x, y, dirs[s], tx, ty);
        x = tx; y = ty;
    }
    Game g2 = g;
    g2.record_events = false;
    g2.move(di, dirs, n);
    const bool alive = g2.dragons[(size_t)di].alive;
#ifdef BC_QD_DEBUG
    if (getenv("BC_QD_DEBUG")) {
        const Dragon& q2 = g2.dragons[(size_t)di];
        fprintf(stderr, "QD round %d alive %d len %d->%d head %d,%d dead_end %d\n", g.round, (int)alive, d.len,
                q2.len, alive ? q2.head() % m.w : -1, alive ? q2.head() / m.w : -1,
                alive ? (int)queen_in_dead_end(g2, di, hx, hy, own_walls) : -1);
    }
#endif
    return (!alive || queen_in_dead_end(g2, di, hx, hy, own_walls)) ? 1 : 0;
}

// VecConfig::queen_deadend. Each legal move (1-3 step paths, far sprints) is played on a copy: fatal
// when it leaves her dead or in a dead end she can see (queen_in_dead_end). A move whose path crosses
// a portal is not judged (the bot cannot see where it lets out) and counts as a way out. Fatal moves
// are dropped only while some legal move is not fatal.
// Level 1: splits and the self-kill are left as they are and are not a way out.
// Level 2: her own body walls the tube where it cannot move off in time (queen_in_dead_end own_walls),
// and when EVERY move is fatal each legal split is tried on a copy: it rescues her if, from there,
// some move she could play (path_legal, heads blocking as in the queen guard) is not fatal. If one does, only the rescuing splits stay
// (fatal moves, stalling splits and the self-kill go); if none does, the mask is left alone.
inline void VecEnv::queen_deadend_mask(const Game& g, int di, uint8_t* mask, int level) {
    const MapData& m = *g.map;
    const Dragon& d = g.dragons[(size_t)di];
    const int hx = d.head() % m.w, hy = d.head() / m.w;
    const bool own_walls = level >= 2;
    uint8_t fatal[CODEC_ACTIONS] = {};
    int n_fatal = 0, n_ok = 0;
    for (int id = 0; id < CODEC_ACTIONS; id++) {
        if (!mask[id]) continue;
        const Action a = decode_full(g, di, id);
        if (a.kind != 0) continue;                         // a split or the self-kill
        char dirs[MAX_STEPS];
        for (int s = 0; s < a.n_steps; s++) dirs[s] = dir_char(a.dirs[s]);
        if (queen_move_fatal(g, di, dirs, a.n_steps, hx, hy, own_walls)) { fatal[id] = 1; n_fatal++; }
        else n_ok++;
    }
    if (n_fatal > 0 && n_ok > 0) {
        for (int id = 0; id < CODEC_ACTIONS; id++)
            if (fatal[id]) mask[id] = 0;
        return;
    }
    if (level < 2 || n_fatal == 0) return;
    // every move is fatal: which splits leave her a move that is not?
    uint8_t rescue[CODEC_ACTIONS] = {};
    int n_rescue = 0;
    for (int id = 0; id < CODEC_ACTIONS; id++) {
        if (!mask[id]) continue;
        const Action a = decode_full(g, di, id);
        if (a.kind != 1) continue;
        Game g2 = g;
        g2.record_events = false;
        g2.split(di, a.split_k);
        if (!g2.dragons[(size_t)di].alive) continue;
        bool out = false;
        for (int mid = 0; mid < CODEC_ACTIONS && !out; mid++) {
            const Action b = decode_full(g2, di, mid);
            if (b.kind != 0 || b.n_steps == 0) continue;
            char dirs[MAX_STEPS];
            for (int s = 0; s < b.n_steps; s++) dirs[s] = dir_char(b.dirs[s]);
            if (path_legal(g2, di, dirs, b.n_steps, true) && !queen_move_fatal(g2, di, dirs, b.n_steps, hx, hy, true))
                out = true;
        }
        if (out) { rescue[id] = 1; n_rescue++; }
    }
    if (n_rescue > 0)
        for (int id = 0; id < CODEC_ACTIONS; id++) mask[id] = mask[id] && rescue[id];
}

// Perfect play (declared in bc_vec.hpp; supervised_learning.md, user 2026-10-02). The kinds,
// first match wins:
//   PP_QUEEN_KILL     a non-queen can kill the enemy queen this move: her head in the window and a
//                     move over open edges (no portal: the dragon sees the whole path) that, played
//                     on a copy of the game, leaves her dead and our team with a dragon alive (a
//                     head-on kills the mover too). Every such move is correct.
//   PP_TRAPPED_QUEEN  a non-queen walls our queen in: every side of her head is kelp, her own body,
//                     or this dragon's body (at least one), she and her four neighbours are in the
//                     window, nobody who moves before her next turn is part of the wall, and no move
//                     of this dragon frees a side (tried on copies). Correct: the self-kill.
//   PP_KEEP_WALL      the same picture for the ENEMY queen, walled in partly by this dragon: every
//                     move or split that, on a copy, keeps her shut and this dragon alive is correct
//                     (user: "don't break a wall"); not labelled when none does, or when all do.
//   PP_QUEEN_DEADEND  (user, 2026-10-05: "no unforced queen error") the acting dragon is a queen and
//                     some legal action leaves her (on a copy) in a dead end she can see: walking on
//                     from her new head there is exactly one way at every tile until there is none,
//                     every tile of it inside the window she decides from, no portal, no dragon beside
//                     it (it might move). She cannot turn round in it, so she dies at its end. Labelled
//                     only when some other legal action is not such a dive: every one of those is
//                     correct. (A non-queen may still split its way out; a queen may not.)
//   PP_LATE_SUICIDE   rounds PP_LATE_FROM..PP_LATE_TO, a non-queen with no enemy in the window and
//                     our queen's head within PP_LATE_RADIUS: the self-kill (its body becomes pearls
//                     beside her, and the queen's length decides the tiebreak).
//   PP_PEARL          no other dragon in the window and a pearl on a tile beside the head (ahead,
//                     left or right, over an open edge): every legal 1-3 step path whose first step
//                     takes it, with no paid step, ending on a tile with a free way on.
//   PP_BLANK          no pearl and no other dragon in the window, the edge ahead open (not kelp or a
//                     portal), the tile ahead free and not a dead end: one step straight on (id 0).
inline int VecEnv::perfect(int env_index, uint8_t* acts) const {
    memset(acts, 0, CODEC_ACTIONS);
    const Env& e = envs_[(size_t)env_index];
    if (e.acting < 0 || e.game.finished) return PP_NONE;
    const Game& g = e.game;
    const MapData& m = *g.map;
    const int di = e.acting;
    const Dragon& d = g.dragons[(size_t)di];
    if (!d.alive) return PP_NONE;
    const uint8_t* mask = b_mask_ + (size_t)env_index * CODEC_ACTIONS;
    const int hx = d.head() % m.w, hy = d.head() / m.w;
    const int facing = dir_index(d.facing);
    static const char DIRS[4] = {'N', 'E', 'S', 'W'};
    // every move id: the 1-3 step paths, then the far sprints
    int move_ids[CODEC_MOVES + CODEC_FAR];
    int n_move_ids = 0;
    for (int id = 0; id < CODEC_MOVES; id++) move_ids[n_move_ids++] = id;
    for (int k = 0; k < CODEC_FAR; k++) move_ids[n_move_ids++] = CODEC_FAR_ID + k;
    auto offset = [&](int16_t cell, int& ddx, int& ddy) {      // the obs code's offset on the torus
        ddx = cell % m.w - hx;
        ddy = cell / m.w - hy;
        if (ddx > m.w / 2) ddx -= m.w;
        if (ddx < -(m.w - 1) / 2) ddx += m.w;
        if (ddy > m.h / 2) ddy -= m.h;
        if (ddy < -(m.h - 1) / 2) ddy += m.h;
    };
    auto queen_of = [&](int team) -> int {                       // her dragon index, -1 when dead
        for (int k = 0; k < 2 && k < (int)g.dragons.size(); k++)
            if (g.dragons[(size_t)k].team == team) return g.dragons[(size_t)k].alive ? k : -1;
        return -1;
    };
    auto edge_kind = [&](int x, int y, char dir) -> uint8_t {
        bool vertical; int ex, ey;
        edge_on_side(m, x, y, dir, vertical, ex, ey);
        const int edge = m.idx(ex, ey);
        return vertical ? m.v_kind[edge] : m.h_kind[edge];
    };
    auto path_of = [&](int id, char* dirs) -> int {
        const Action a = decode_full(g, di, id);
        if (a.kind != 0) return 0;
        for (int s = 0; s < a.n_steps; s++) dirs[s] = dir_char(a.dirs[s]);
        return a.n_steps;
    };
    auto play = [&](int id) -> Game {                            // the action on a copy of the game
        Game g2 = g;
        g2.record_events = false;
        const Action a = decode_full(g, di, id);
        if (a.kind == 0) {
            char dirs[MAX_STEPS];
            for (int s = 0; s < a.n_steps; s++) dirs[s] = dir_char(a.dirs[s]);
            g2.move(di, dirs, a.n_steps);
        } else if (a.kind == 1) {
            g2.split(di, a.split_k);
        } else {
            g2.kill(di, DEATH_ACTION);
        }
        return g2;
    };
    // A queen's free sides in gg: by_me counts sides walled by this dragon; unsure is set by a
    // portal (its far side is not in the window) or by a wall that may move before she does
    // (later this round, or before her next round). Her own body, kelp and this dragon are sure.
    auto free_sides = [&](const Game& gg, int q, int& by_me, bool& unsure) {
        const int16_t qh = g.dragons[(size_t)q].head();
        const int qx0 = qh % m.w, qy0 = qh / m.w;
        int n_free = 0;
        by_me = 0;
        unsure = false;
        for (char dir : DIRS) {
            const uint8_t kind = edge_kind(qx0, qy0, dir);
            if (kind == EDGE_KELP) continue;
            if (kind == EDGE_PORTAL) { unsure = true; continue; }
            int nx, ny;
            tile_after_step(m, qx0, qy0, dir, nx, ny);
            const int16_t occ = gg.owner[m.idx(nx, ny)];
            if (occ < 0) n_free++;
            else if (occ == (int16_t)q) continue;
            else if (occ == (int16_t)di) by_me++;
            else if (occ > di || occ < q) unsure = true;
        }
        return n_free;
    };
    auto inner = [&](int q) {             // her head and its four neighbours are in the window
        int qx, qy;
        offset(g.dragons[(size_t)q].head(), qx, qy);
        return std::abs(qx) <= VISION - 1 && std::abs(qy) <= VISION - 1;
    };
    const bool me_queen = Game::is_queen(d);
    const int our_q = queen_of(d.team), their_q = queen_of(1 - d.team);

    // the window, walked as the obs code walks it
    bool pearl = false, ally = false, foe = false, their_q_head = false;
    for (int row = 0; row < WINDOW; row++)
        for (int col = 0; col < WINDOW; col++) {
            int wx, wy;
            ego_to_world(facing, col - VISION, row - VISION, wx, wy);
            const int t = m.idx(m.wrapx(hx + wx), m.wrapy(hy + wy));
            if (g.pearl[t]) pearl = true;
            const int16_t occ = g.owner[t];
            if (occ < 0 || occ == (int16_t)di) continue;
            if (g.dragons[(size_t)occ].team == d.team) {
                ally = true;
            } else {
                foe = true;
                if (occ == their_q && g.head_at[t]) their_q_head = true;
            }
        }

    // ---- the enemy queen within one move
    if (!me_queen && their_q >= 0 && their_q_head) {
        bool any = false;
        for (int j = 0; j < n_move_ids; j++) {
            const int id = move_ids[j];
            if (!mask[id]) continue;
            char dirs[MAX_STEPS];
            const int n = path_of(id, dirs);
            if (n == 0) continue;
            int x = hx, y = hy;
            bool open = true;
            for (int s = 0; s < n && open; s++) {
                if (edge_kind(x, y, dirs[s]) != EDGE_OPEN) { open = false; break; }
                int nx, ny;
                tile_after_step(m, x, y, dirs[s], nx, ny);
                x = nx; y = ny;
            }
            if (!open) continue;
            const Game g2 = play(id);
            // and not with our last dragon: a head-on kills both, and losing every dragon loses
            // the game (or draws it, when she was their last) whatever the queens
            if (!g2.dragons[(size_t)their_q].alive && g2.alive[d.team] > 0) { acts[id] = 1; any = true; }
        }
        if (any) return PP_QUEEN_KILL;
    }

    // ---- our queen walled in by this dragon
    if (CODEC_SUICIDE && !me_queen && our_q >= 0 && inner(our_q)) {
        int by_me;
        bool unsure;
        if (free_sides(g, our_q, by_me, unsure) == 0 && by_me > 0 && !unsure) {
            bool a_move_frees = false;
            for (int j = 0; j < n_move_ids && !a_move_frees; j++) {
                const int id = move_ids[j];
                if (!mask[id]) continue;
                const Game g2 = play(id);
                int b2;
                bool u2;
                if (g2.dragons[(size_t)our_q].alive && free_sides(g2, our_q, b2, u2) > 0) a_move_frees = true;
            }
            if (!a_move_frees) {
                acts[CODEC_SUICIDE_ID] = 1;
                return PP_TRAPPED_QUEEN;
            }
        }
    }

    // A queen in gg (dragon di) in a dead end she can see from where she decides (hx, hy): one way
    // on at every tile until none. A portal, a dragon beside the tube or a tile out of the window
    // means not sure, so not a dead end.
    auto dead_end = [&](const Game& gg) -> bool {
        const Dragon& q = gg.dragons[(size_t)di];
        if (!q.alive) return false;
        int x = q.head() % m.w, y = q.head() / m.w;
        char came = q.facing;
        for (int depth = 0; depth <= 2 * WINDOW; depth++) {
            int ox, oy;
            offset((int16_t)m.idx(x, y), ox, oy);
            if (std::abs(ox) > VISION || std::abs(oy) > VISION) return false;
            const char back = Game::opposite(came);
            int ways = 0, nx = 0, ny = 0;
            char nd = 0;
            for (char dir : DIRS) {
                if (dir == back) continue;
                const uint8_t kind = edge_kind(x, y, dir);
                if (kind == EDGE_KELP) continue;
                if (kind == EDGE_PORTAL) return false;
                int tx, ty;
                tile_after_step(m, x, y, dir, tx, ty);
                if (gg.owner[m.idx(tx, ty)] >= 0) return false;
                ways++;
                nx = tx; ny = ty; nd = dir;
            }
            if (ways == 0) return true;
            if (ways > 1) return false;
            x = nx; y = ny; came = nd;
        }
        return false;
    };

    // ---- a queen about to dive into a dead end she can see, with a way not to
    if (me_queen) {
        uint8_t ok[CODEC_ACTIONS] = {};
        int n_err = 0, n_ok = 0;
        for (int id = 0; id < CODEC_ACTIONS; id++) {
            if (!mask[id]) continue;
            const Game g2 = play(id);
            if (!g2.dragons[(size_t)di].alive) continue;
            if (dead_end(g2)) n_err++;
            else { ok[id] = 1; n_ok++; }
        }
        if (n_err > 0 && n_ok > 0) {
            memcpy(acts, ok, CODEC_ACTIONS);
            return PP_QUEEN_DEADEND;
        }
    }

    // ---- the enemy queen walled in, partly by this dragon: keep her shut
    if (their_q >= 0 && inner(their_q)) {
        int by_me;
        bool unsure;
        if (free_sides(g, their_q, by_me, unsure) == 0 && by_me > 0 && !unsure) {
            int n_ok = 0, n_legal = 0;
            for (int id = 0; id < CODEC_ACTIONS; id++) {
                if (!mask[id]) continue;
                n_legal++;
                const Game g2 = play(id);
                if (!g2.dragons[(size_t)di].alive || !g2.dragons[(size_t)their_q].alive) continue;
                int b2;
                bool u2;
                if (free_sides(g2, their_q, b2, u2) == 0 && !u2) { acts[id] = 1; n_ok++; }
            }
            if (n_ok > 0 && n_ok < n_legal) return PP_KEEP_WALL;
            memset(acts, 0, CODEC_ACTIONS);
        }
    }

    // ---- the last rounds, beside our queen, no enemy in sight
    if (CODEC_SUICIDE && !me_queen && our_q >= 0 && !foe &&
        g.round >= PP_LATE_FROM && g.round <= PP_LATE_TO) {
        int qx, qy;
        offset(g.dragons[(size_t)our_q].head(), qx, qy);
        if (std::max(std::abs(qx), std::abs(qy)) <= PP_LATE_RADIUS) {
            acts[CODEC_SUICIDE_ID] = 1;
            return PP_LATE_SUICIDE;
        }
    }

    // a free tile beyond (x, y), over an open edge, other than straight back the way `came`
    auto way_on = [&](const Game& gg, int x, int y, char came) {
        const char back = Game::opposite(came);
        for (char dir : DIRS) {
            if (dir == back || edge_kind(x, y, dir) != EDGE_OPEN) continue;
            int nx, ny;
            tile_after_step(m, x, y, dir, nx, ny);
            if (gg.owner[m.idx(nx, ny)] < 0) return true;
        }
        return false;
    };

    // ---- a pearl beside the head, nobody else in sight
    if (pearl && !ally && !foe) {
        bool any = false;
        const int free = Game::free_steps(d.len);
        for (int id = 0; id < CODEC_MOVES; id++) {
            if (!mask[id]) continue;
            char dirs[MAX_STEPS];
            const int n = path_of(id, dirs);
            if (n == 0 || n > free) continue;                 // no paid step
            if (edge_kind(hx, hy, dirs[0]) != EDGE_OPEN) continue;
            int px, py;
            tile_after_step(m, hx, hy, dirs[0], px, py);
            if (!g.pearl[m.idx(px, py)]) continue;
            const Game g2 = play(id);
            const Dragon& d2 = g2.dragons[(size_t)di];
            if (!d2.alive) continue;
            if (!way_on(g2, d2.head() % m.w, d2.head() / m.w, d2.facing)) continue;
            if (me_queen && dead_end(g2)) continue;              // never a queen's dive
            acts[id] = 1;
            any = true;
        }
        if (any) return PP_PEARL;
    }

    // ---- an empty window: straight on
    if (!pearl && !ally && !foe && mask[0] && edge_kind(hx, hy, d.facing) == EDGE_OPEN) {
        int fx, fy;
        tile_after_step(m, hx, hy, d.facing, fx, fy);
        if (g.owner[m.idx(fx, fy)] < 0 && way_on(g, fx, fy, d.facing)) {
            acts[0] = 1;
            return PP_BLANK;
        }
    }
    return PP_NONE;
}

// A synthetic position (declared in bc_vec.hpp).
inline bool VecEnv::set_scenario(int env_index, int round, int n, const int* team, const int* len,
                                 const int* cells, int n_pearls, const int* pearls, int acting) {
    Env& e = envs_[(size_t)env_index];
    begin_episode(e, env_index);
    Game& g = e.game;
    const MapData& m = *e.map;
    const int area = m.area();
    // check everything before touching the game
    std::vector<char> used((size_t)area, 0);
    int at = 0;
    for (int j = 0; j < n; j++) {
        if (team[j] != 0 && team[j] != 1) return false;
        for (int i = 0; i < len[j]; i++) {
            const int c = cells[at + i];
            if (c < 0 || c >= area || used[(size_t)c]) return false;
            used[(size_t)c] = 1;
            if (i > 0) {
                const int b = cells[at + i - 1];
                if (!direction_between(m, c % m.w, c / m.w, b % m.w, b / m.w)) return false;
            }
        }
        at += len[j];
    }
    if (acting < 0 || acting >= n || len[acting] <= 0) return false;
    for (int i = 0; i < n_pearls; i++)
        if (pearls[i] < 0 || pearls[i] >= area || used[(size_t)pearls[i]]) return false;

    g.dragons.clear();
    std::fill(g.owner.begin(), g.owner.end(), (int16_t)-1);
    std::fill(g.head_at.begin(), g.head_at.end(), (uint8_t)0);
    std::fill(g.seg_dir.begin(), g.seg_dir.end(), (uint8_t)0);
    std::fill(g.pearl.begin(), g.pearl.end(), (uint8_t)0);
    g.alive[0] = g.alive[1] = 0;
    at = 0;
    for (int j = 0; j < n; j++) {
        if (len[j] <= 0) {
            g.add_dead(team[j]);
        } else {
            DragonSpawn sp;
            sp.team = team[j];
            for (int i = 0; i < len[j]; i++) sp.body.push_back((int16_t)cells[at + i]);
            // a one-segment dragon faces a random way (a longer one: its neck to its head)
            g.add_spawn(sp, dir_char((int)(e.rng() % 4)));
        }
        at += std::max(len[j], 0);
    }
    for (int i = 0; i < n_pearls; i++) g.pearl[(size_t)pearls[i]] = 1;
    g.round = round;
    for (Dragon& d : g.dragons) d.protocol = 3;
    e.agents.assign(g.dragons.size(), AgentAcc());
    for (size_t i = 0; i < e.agents.size(); i++) {
        e.agents[i].uid = e.uid_of(g.dragons[i].id);
        e.agents[i].last_len = g.dragons[i].len;
    }
    e.mem_clear();
    e.s2_last.clear();
    e.round_open = true;
    e.cursor = acting;
    e.acting = acting;
    e.agents[(size_t)acting].open = true;
    observe(e, env_index);
    return true;
}

}  // namespace bc
