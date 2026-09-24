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
                else if (o.team == d.team) at(head ? LC_ALLY_HEAD : LC_ALLY_BODY, row, col) = 1.0f;
                else {
                    at(head ? LC_ENEMY_HEAD : LC_ENEMY_BODY, row, col) = 1.0f;
                    dm.saw_foe(x, y, g.round);   // only the wide planes read this
                }
                const int world_dir = g.seg_dir[t];
                const int shown = (world_dir - facing + 4) % 4;
                at(LC_FACE_N + shown, row, col) = 1.0f;
            }

            bool kelp_here = false;
            for (int o_dir = 0; o_dir < 4; o_dir++) {
                const int world_dir = (o_dir + facing) % 4;
                bool vertical; int ex, ey;
                edge_on_side(m, x, y, dir_char(world_dir), vertical, ex, ey);
                const int edge = m.idx(ex, ey);
                const uint8_t kind = vertical ? m.v_kind[edge] : m.h_kind[edge];
                if (kind == EDGE_KELP) { at(LC_KELP_N + o_dir, row, col) = 1.0f; kelp_here = true; }
                else if (kind == EDGE_PORTAL) at(LC_PORTAL_N + o_dir, row, col) = 1.0f;
            }
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
    sc[SC_NUM_MSGS] = (float)std::min<size_t>(d.inbox.size(), MAX_MSGS);
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

    // What this dragon's own sonars came back with last turn. Zero throughout
    // unless the env was built with sonar on, because nothing is cast then.
    for (int k = 0; k < SONAR_ECHO_KINDS; k++)
        sc[SC_ECHO_AT + k] = (float)d.echo[k] / (float)SONAR_DIRS;

    uint32_t* msgs = b_msgs_ + (size_t)index * MAX_MSGS;
    for (int i = 0; i < MAX_MSGS; i++)
        msgs[i] = i < (int)d.inbox.size() ? d.inbox[i] : 0u;

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

    if (b_board_) {
        const size_t plane = (size_t)BOARD_MAX * BOARD_MAX;
        uint8_t* bd = b_board_ + (size_t)index * BOARD_CH * plane;
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
                }
                bd[4 * plane + at] = g.pearl[t];
                bd[5 * plane + at] = 1;
                bd[6 * plane + at] = m.h_kind[t] == EDGE_KELP;
                bd[7 * plane + at] = m.v_kind[t] == EDGE_KELP;
                // When a pearl is due here, as nearness rather than delay:
                // 255 is due now, small is far off, 0 is never. A countdown
                // past 255 rounds is beyond any game, so it reads as never.
                bd[10 * plane + at] = g.cd[t] < 0 ? 0
                    : (uint8_t)(255 - std::min(g.cd[t], 255));
            }
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
inline void VecEnv::fill_mask(Env& e, int index) {
    const Game& g = e.game;
    const MapData& m = *e.map;
    const int di = e.acting;
    const Dragon& d = g.dragons[di];
    uint8_t* mask = b_mask_ + (size_t)index * CODEC_ACTIONS;
    memset(mask, 0, CODEC_ACTIONS);

    int16_t added[3];
    for (int id = 0; id < CODEC_MOVES; id++) {
        const Action a = decode_for(d, id);
        int x = d.head() % m.w, y = d.head() / m.w;
        int length = d.len, dropped = 0, n_added = 0;
        bool legal = true;
        for (int s = 0; s < a.n_steps; s++) {
            if (s > 0 && length <= 2) { legal = false; break; }
            const char dir = dir_char(a.dirs[s]);
            bool vertical; int ex, ey;
            edge_on_side(m, x, y, dir, vertical, ex, ey);
            const int edge = m.idx(ex, ey);
            const uint8_t kind = vertical ? m.v_kind[edge] : m.h_kind[edge];
            if (kind == EDGE_KELP) { legal = false; break; }
            if (kind == EDGE_PORTAL) break;      // cannot see the far side
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
            if (!blocked && g.owner[t] >= 0 && g.owner[t] != (int16_t)di && !g.head_at[t])
                blocked = true;                   // another dragon's body is certain death
            if (blocked) { legal = false; break; }

            if (n_added < 3) added[n_added++] = (int16_t)t;
            if (g.pearl[t]) length++;
            else dropped++;
            if (s > 0) { dropped++; length--; }
            x = nx; y = ny;
        }
        mask[id] = legal ? 1 : 0;
    }

    const bool room = g.alive[d.team] < m.unit_limit;
    for (int i = 0; i < CODEC_SPLITS; i++) {
        const int k = CODEC_SPLIT_K[i] < 0 ? d.len / 2 : CODEC_SPLIT_K[i];
        mask[CODEC_MOVES + i] = (room && k >= 2 && k <= d.len - 2) ? 1 : 0;
    }
}

}  // namespace bc
