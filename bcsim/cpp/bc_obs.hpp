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
                else at(head ? LC_ENEMY_HEAD : LC_ENEMY_BODY, row, col) = 1.0f;
                const int world_dir = g.seg_dir[t];
                const int shown = (world_dir - facing + 4) % 4;
                at(LC_FACE_N + shown, row, col) = 1.0f;
            }

            for (int o_dir = 0; o_dir < 4; o_dir++) {
                const int world_dir = (o_dir + facing) % 4;
                bool vertical; int ex, ey;
                edge_on_side(m, x, y, dir_char(world_dir), vertical, ex, ey);
                const int edge = m.idx(ex, ey);
                const uint8_t kind = vertical ? m.v_kind[edge] : m.h_kind[edge];
                if (kind == EDGE_KELP) at(LC_KELP_N + o_dir, row, col) = 1.0f;
                else if (kind == EDGE_PORTAL) at(LC_PORTAL_N + o_dir, row, col) = 1.0f;
            }
        }

    float* sc = b_scalar_ + (size_t)index * SC_COUNT;
    memset(sc, 0, sizeof(float) * SC_COUNT);
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

    uint32_t* msgs = b_msgs_ + (size_t)index * MAX_MSGS;
    for (int i = 0; i < MAX_MSGS; i++)
        msgs[i] = i < (int)d.inbox.size() ? d.inbox[i] : 0u;

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
