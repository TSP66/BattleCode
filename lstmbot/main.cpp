// The FFL policy bot (bcsim/train/ff_net.py FFLPolicy), one process per dragon.
//
// What the s2g15p simulator (libbcvec_*_s2_g15p: -DBC_SONAR2 -DBC_XSPLIT -DBC_GRID_G=15
// -DBC_PORTALREP -DBC_FARSPRINT) shows a policy, rebuilt from the protocol block each turn:
//
//   memory     bc_memory.hpp's DragonMemory, the simulator's own code (copied verbatim), updated
//              in the order VecEnv::observe updates it: the 7x7 window in the dragon's frame (cells,
//              edges, allies, foes, queens), where the head stands, the verified team packets in
//              inbox order, and the dragon's own last portal;
//   grid       the 57 x 15 x 15 grid in float, as bc_obs.hpp draws it (memory planes from
//              DragonMemory::grid, the live window, the tracked body, the constant planes, the
//              reporters' lengths, the portal reports, the queens, the dragon's identity), then
//              Q12 for the CNN, which reads the first 54 channels;
//   mlp        mlp0 reads [LayerNorm(embed, previous action), 5 scalars, 3 identity scalars,
//              75 action-history inputs, a zero pad to an even width] (ff_net.py scal_feats /
//              ident_feats off the grid's constant planes; ahist_feats: VecEnv::note_action's history);
//   mask       75 actions (obs.hpp): the 1-3 step paths, the splits, the self-kill, the two
//              splits from the end, the 24 far sprints, and the queen guard (BC_QUEEN_GUARD=1);
//   packet     the sonar v2 packet with the portal report (bc_sonar2.hpp, BC_PORTALREP), as
//              VecEnv::s2_packet writes it, cast four ways with protocol 3.
//
// The network (lstmnet.hpp kernels, weights from bcsim/train/export_ffl.py, the temperature folded
// in at the gate's greedy temperature) plays the masked argmax, as the gate does.
//
// -DBC_DUMP (parity builds: native, or wasm under wasmprobe/wasmrun.py) writes, per turn, to
// $BC_DUMP_FILE (else stderr): the
// previous action, the CNN's grid as Q12 integers, the mlp's 8 scalars, the mask, the logits and
// the action; and prints "ACT <id>" on stdout before ENDTURN, so a harness can step its simulator.
#define BC_SONAR2 1
#define BC_PORTALREP 1
#define BC_GRID_G 15

// the queen dead-end mask's level (obs.hpp Snapshot::queen_deadend; the simulator's BC_QUEEN_DEADEND)
#ifndef BC_QD_LEVEL
#define BC_QD_LEVEL 2
#endif
#include "helper.hpp"
#include "lstmnet.hpp"
#include "obs.hpp"
#include "bc_sonar2.hpp"
#include "bc_memory.hpp"
#include "weights.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace {

using ln::i16;
using ln::i32;
namespace gc = bc::grid_cfg;

constexpr int G = gc::G, GC = gc::CELLS;    // 15, 225
constexpr int HALF = gc::HALF;              // head at row 7, col 7
constexpr int GP = (GC + 3) / 4 * 4;        // 228: planes padded to a multiple of 4 (the wasm gemm)
constexpr int D = (G + 1) / 2, DC = D * D;  // after the stride-2 conv: 8x8 = 64
constexpr int NCH = wt::IN_CH;              // 54: the CNN reads channels 0..53
constexpr int N_ACT = wt::N_ACT;
constexpr int NO_ACTION = N_ACT;
constexpr int PREV_DIM = 48;
constexpr int MAX_ROUNDS = 500;             // the simulator's cfg max_rounds, and the judge's
constexpr int XIN = wt::EMB + PREV_DIM;     // 304, what the LayerNorm reads
constexpr int N_SCAL = 5, N_IDENT = 3, N_AHIST = N_ACT;
constexpr int MLP_X = (wt::SCAL_IN ? N_SCAL : 0) + (wt::IDENT_IN ? N_IDENT : 0) + (wt::AHIST_IN ? N_AHIST : 0);
constexpr int MLP_IN = (XIN + MLP_X + 1) / 2 * 2;   // export_ffl.py mlp0_width: an odd width gets a zero column
constexpr int CENTRE = HALF * G + HALF;
static_assert(wt::GRID == G, "weights for another grid size");
static_assert(N_ACT == obs::N_ACTIONS, "weights for another action codec");
static_assert(NCH <= gc::BIRTH && NCH > gc::IS_QUEEN, "the CNN reads channels 0..53");
static_assert(MLP_IN == wt::MLP_IN, "mlp0's input width");
static_assert(DC % 4 == 0 && wt::HID % 4 == 0 && MLP_IN % 2 == 0);
// scal_in / ident_in: the grid channels they read (train/ff_net.py SCAL_* / IDENT_BIRTH)
static_assert(gc::ROUND == 28 && gc::LENGTH == 29 && gc::UNITS == 30 && gc::IS_QUEEN == 43 && gc::BIRTH == 54);

// the judge's virtual clock counts CPU points, so this is the bot's own meter
std::int64_t points_now() {
    auto const t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

inline int wrap(int v, int m) { return (v % m + m) % m; }

// ---------------------------------------------------------------- weights
std::vector<i16> WB;
std::size_t at = 0, fat = 0;
i16 const* take(std::size_t n) { i16 const* p = WB.data() + at; at += n; return p; }
float const* ftake(std::size_t n) { float const* p = wt::FP + fat; fat += n; return p; }

struct Layer { i16 const* w; float const *s, *b, *g = nullptr, *beta = nullptr; int cin, cout, k; };
Layer conv(int cin, int cout, int k, bool norm) {
    Layer l{};
    l.cin = cin; l.cout = cout; l.k = k;
    l.w = take((std::size_t)cout * cin * k * k);
    l.s = ftake((std::size_t)cout);
    l.b = ftake((std::size_t)cout);
    if (norm) { l.g = ftake((std::size_t)cout); l.beta = ftake((std::size_t)cout); }
    return l;
}
Layer lin(int in, int out) {
    Layer l{};
    l.cin = in; l.cout = out; l.k = 1;
    l.w = take((std::size_t)in * out);
    l.s = ftake((std::size_t)out);
    l.b = ftake((std::size_t)out);
    return l;
}

Layer stem, down, squeeze, embed, pi;
std::vector<Layer> res15, res8, mlp;
// The wasm gemm needs a multiple of 4 outputs (lstmnet.hpp); 75 actions are not, so the policy
// layer is repacked at load to PI_P columns, the extra ones zero.
constexpr int PI_P = (N_ACT + 3) / 4 * 4;
std::vector<i16> pi_pad;
float const *prev_table, *ln_g, *ln_b;

bool load() {
    ln::build_silu();
    WB.resize(wt::COUNT + 16);
    ln::join_planes(wt::HI, wt::LO, wt::COUNT, WB.data());
    stem = conv(NCH, wt::C1, 3, true);
    for (int i = 0; i < 2 * wt::B1; i++) res15.push_back(conv(wt::C1, wt::C1, 3, true));
    down = conv(wt::C1, wt::C2, 3, true);
    for (int i = 0; i < 2 * wt::B2; i++) res8.push_back(conv(wt::C2, wt::C2, 3, true));
    squeeze = conv(wt::C2, wt::SQ, 1, false);
    embed = lin(wt::SQ * DC, wt::EMB);
    for (int l = 0; l < wt::LAYERS; l++) mlp.push_back(lin(l == 0 ? MLP_IN : wt::HID, wt::HID));
    pi = lin(wt::HID, N_ACT);
    if (PI_P != N_ACT) {                         // [K/2][P][2] -> [K/2][PI_P][2]
        pi_pad.assign((std::size_t)(wt::HID / 2) * PI_P * 2 + 16, 0);
        for (int p = 0; p < wt::HID / 2; p++)
            for (int j = 0; j < N_ACT; j++)
                for (int t = 0; t < 2; t++)
                    pi_pad[((std::size_t)p * PI_P + j) * 2 + t] = pi.w[((std::size_t)p * N_ACT + j) * 2 + t];
        pi.w = pi_pad.data();
    }
    prev_table = ftake((std::size_t)(N_ACT + 1) * PREV_DIM);
    ln_g = ftake(XIN);
    ln_b = ftake(XIN);
    return at == wt::COUNT && fat == wt::N_FP;
}

// ---------------------------------------------------------------- memory
// The simulator's own DragonMemory (bc_memory.hpp), fed what the protocol shows.
bc::DragonMemory dm;
int W_ = 0, H_ = 0;
int my_id = 0, birth = -1;
float qwin[2][obs::CELLS];                   // the queens' segments in the window (0 ours, 1 theirs)

// ego offset -> world offset, per facing (N, E, S, W): bc_obs.hpp ego_to_world
inline void ego_to_world(int f, int ox, int oy, int& wx, int& wy) {
    switch (f) {
        case 0: wx = ox; wy = oy; break;
        case 1: wx = -oy; wy = ox; break;
        case 2: wx = -ox; wy = -oy; break;
        default: wx = oy; wy = -ox; break;
    }
}
inline void world_to_ego(int f, int wx, int wy, int& ox, int& oy) {
    switch (f) {
        case 0: ox = wx; oy = wy; break;
        case 1: ox = wy; oy = -wx; break;
        case 2: ox = -wx; oy = -wy; break;
        default: ox = -wy; oy = wx; break;
    }
}
// the torus offset from the head, as bc_obs.hpp computes it
inline void torus(int& ddx, int& ddy) {
    if (ddx > W_ / 2) ddx -= W_;
    if (ddx < -(W_ - 1) / 2) ddx += W_;
    if (ddy > H_ / 2) ddy -= H_;
    if (ddy < -(H_ - 1) / 2) ddy += H_;
}

// The window into memory, in VecEnv::observe's order: its own queen sighting, then every window
// cell in the dragon's frame (row-major), then where the head stands.
void observe(obs::Snapshot const& s, int round) {
    std::memset(qwin, 0, sizeof(qwin));
    if (my_id < 2) dm.saw_queen(0, s.hx, s.hy, round, true);     // a queen knows where it is
    for (int row = 0; row < obs::WINDOW; row++)
        for (int col = 0; col < obs::WINDOW; col++) {
            int wx, wy;
            ego_to_world(s.facing, col - obs::VISION, row - obs::VISION, wx, wy);
            int const x = wrap(s.hx + wx, W_), y = wrap(s.hy + wy, H_);
            auto const& t = s.tile((wy + obs::VISION) * obs::WINDOW + (wx + obs::VISION));
            if (auto const* part = t.get_dragon()) {
                bool const same = (char)part->team.value == s.my_team;
                if (part->dragon_id != my_id) {
                    if (same) dm.saw_ally(x, y, round);
                    else dm.saw_foe(x, y, round);
                    if (part->dragon_id < 2) {                       // ids 0 and 1 are the queens
                        int const side = same ? 0 : 1;
                        qwin[side][row * obs::WINDOW + col] = 1.0f;
                        dm.saw_queen(side, x, y, round, part->is_dragon_head, bc::s2::bearing(wx, wy));
                    }
                }
            }
            bool kelp_here = false;
            std::uint8_t sides = 0;                  // world directions
            for (int o_dir = 0; o_dir < 4; o_dir++) {
                int const world_dir = (o_dir + s.facing) % 4;
                auto const kind = t.edges[(std::size_t)world_dir].edge_type;
                if (kind == unswbc::EdgeType::KELP) {
                    kelp_here = true;
                    sides |= (std::uint8_t)(1 << world_dir);
                } else if (kind == unswbc::EdgeType::PORTAL) {
                    sides |= (std::uint8_t)(1 << (4 + world_dir));
                }
            }
            dm.see_sides(x, y, sides);
            dm.see(x, y, round, t.pearl, t.pearl_time, kelp_here);
        }
    dm.stand(s.hx, s.hy, round);
}

// ---------------------------------------------------------------- sonar v2
// This turn's verified team packets, in inbox order (VecEnv::observe, BC_SONAR2): each goes into
// memory -- the reporter, its enemy sighting, the queens, its last portal.
constexpr int MAX_MSGS = 64;                 // bc_vec.hpp MAX_MSGS
std::vector<bc::s2::Packet> reports;
std::uint64_t last_sent = 0;                 // our own packet, to drop it when it comes back
bool sent_any = false;

void hear(unswbc::Controller const& ct, obs::Snapshot const& s, int team, int round) {
    reports.clear();
    for (std::uint64_t const v : ct.get_sonar_messages()) {
        if ((int)reports.size() >= MAX_MSGS) break;
        bc::s2::Packet p;
        int sent = -1;
        if (!bc::s2::decode(v, team, round, p, sent)) continue;
        if (p.hx >= W_ || p.hy >= H_) continue;
        if (sent_any && bc::s2::same_sender(v, last_sent)) continue;     // our own ray
        bool dup = false;
        for (auto const& q : reports) dup |= q.hx == p.hx && q.hy == p.hy;
        if (dup) continue;                                               // four rays, one sender
        dm.heard_ally(p.hx, p.hy, sent);
        if (p.enemy && p.ex < W_ && p.ey < H_) dm.heard_foe(p.ex, p.ey, sent);
        if (p.queen) dm.heard_queen(0, p.hx, p.hy, sent);
        if (p.enemy_queen && p.ex < W_ && p.ey < H_) {
            int sx = s.hx - p.hx, sy = s.hy - p.hy;      // the sender to this dragon, on the torus
            torus(sx, sy);
            // a relayed report that came from this dragon's side is its own news coming back
            bool const echo = p.queen_age > 0 && bc::s2::bearing(sx, sy) == p.came_from;
            if (!echo) dm.heard_queen(1, p.ex, p.ey, sent - p.queen_age, bc::s2::bearing(-sx, -sy));
        }
        // a teammate's last portal: kept unless this dragon has seen that entrance and it has no
        // portal on that side (then the packet is a forgery or a tag collision)
        if (p.portal && p.px < W_ && p.py < H_ && !dm.knows_no_portal(p.px, p.py, p.pdir))
            dm.heard_portal(p.px, p.py, p.pdir, p.pbox, p.ppearls, sent);
        reports.push_back(p);
    }
    // its own last portal, as it would report it
    int tiles = 0, pearls = 0;
    if (dm.trip_far_side(round, bc::s2::PORTAL_BOX_MAX, bc::s2::PORTAL_PEARL_STEPS, tiles, pearls))
        dm.heard_portal(dm.trip.ax, dm.trip.ay, dm.trip.dir, bc::s2::portal_box_bucket(tiles),
                        pearls < 3 ? pearls : 3, round);
}

// The packet, as VecEnv::s2_packet writes it: this dragon before its move.
std::uint64_t packet(obs::Snapshot const& s, int team, int round) {
    bc::s2::Packet p;
    p.hx = s.hx;
    p.hy = s.hy;
    p.len = s.length;
    p.facing = s.facing;
    p.queen = my_id < 2;
    // in its own window (world offsets, row-major): the enemy queen if any of it shows (its head,
    // else its nearest segment); else, relayed, where it was last known if at most RELAY_ROUNDS
    // old; else the nearest enemy head -- Chebyshev on the torus
    int best = 99, best_q = 99;
    for (int oy = -obs::VISION; oy <= obs::VISION; oy++)
        for (int ox = -obs::VISION; ox <= obs::VISION; ox++) {
            auto const& t = s.tile((oy + obs::VISION) * obs::WINDOW + (ox + obs::VISION));
            auto const* part = t.get_dragon();
            if (!part || (char)part->team.value == s.my_team) continue;
            int const dist = std::max(std::abs(ox), std::abs(oy));
            if (part->dragon_id < 2) {
                int const rank = part->is_dragon_head ? -1 : dist;     // its head beats any segment
                if (rank < best_q) {
                    best_q = rank;
                    p.enemy = p.enemy_queen = true;
                    p.ex = t.position.x;
                    p.ey = t.position.y;
                }
            } else if (part->is_dragon_head && best_q == 99 && dist < best) {
                best = dist;
                p.enemy = true;
                p.ex = t.position.x;
                p.ey = t.position.y;
            }
        }
    if (best_q == 99) {
        auto const& q = dm.queen[1];
        if (q.round > bc::mem_cfg::NEVER && round - q.round <= bc::s2::RELAY_ROUNDS) {
            p.enemy = p.enemy_queen = true;
            p.ex = q.x;
            p.ey = q.y;
            p.queen_age = round - q.round;
            p.came_from = q.from;
        }
    }
    int tiles = 0, pearls = 0;
    if (dm.trip_far_side(round, bc::s2::PORTAL_BOX_MAX, bc::s2::PORTAL_PEARL_STEPS, tiles, pearls)) {
        p.portal = true;
        p.px = dm.trip.ax;
        p.py = dm.trip.ay;
        p.pdir = dm.trip.dir;
        p.pbox = bc::s2::portal_box_bucket(tiles);
        p.ppearls = pearls < 3 ? pearls : 3;
    }
    return bc::s2::encode(p, team, round);
}

// ---------------------------------------------------------------- portal trips
// DragonMemory::trip is the last portal this dragon went through: the tile it stepped from, the
// direction, and the tile it came out on. The simulator records it when a move crosses exactly one
// portal and the dragon lives. The bot knows the entrance from its own move (the steps up to the
// first portal are in its window); the exit it finds next turn, walking its new head back over the
// steps after the portal. A step that arrived through a portal leaves a portal edge behind the tile
// it arrived on (a portal keeps the heading), which is how a second crossing shows.
struct Pending {
    bool active = false;
    int ax = 0, ay = 0, dir = 0, n_rest = 0;
    int rest[obs::FAR_MAX_STEPS] = {};
} pending;

void plan_trip(obs::Snapshot const& s, int const* dirs, int n) {
    pending.active = false;
    int x = s.hx, y = s.hy;
    for (int i = 0; i < n; i++) {
        int const cell = s.cell_of(x, y);
        if (cell < 0) return;
        auto const kind = s.tile(cell).edges[(std::size_t)dirs[i]].edge_type;
        if (kind == unswbc::EdgeType::KELP) return;        // it dies
        if (kind == unswbc::EdgeType::PORTAL) {
            pending.active = true;
            pending.ax = x;
            pending.ay = y;
            pending.dir = dirs[i];
            pending.n_rest = 0;
            for (int j = i + 1; j < n; j++) pending.rest[pending.n_rest++] = dirs[j];
            return;
        }
        x = wrap(x + obs::DX[(std::size_t)dirs[i]], W_);
        y = wrap(y + obs::DY[(std::size_t)dirs[i]], H_);
    }
}

void land_trip(obs::Snapshot const& s) {
    if (!pending.active) return;
    pending.active = false;
    int x = s.hx, y = s.hy;
    for (int j = pending.n_rest - 1; j >= 0; j--) {
        int const d = pending.rest[j], back = (d + 2) & 3;
        int const cell = s.cell_of(x, y);
        if (cell < 0) return;
        if (s.tile(cell).edges[(std::size_t)back].edge_type == unswbc::EdgeType::PORTAL) return;   // a 2nd crossing
        x = wrap(x - obs::DX[(std::size_t)d], W_);
        y = wrap(y - obs::DY[(std::size_t)d], H_);
    }
    int const cell = s.cell_of(x, y);
    if (cell < 0 || s.tile(cell).edges[(std::size_t)((pending.dir + 2) & 3)].edge_type != unswbc::EdgeType::PORTAL)
        return;                                            // not where a portal lets out: do not guess
    dm.trip = bc::DragonMemory::Trip{pending.ax, pending.ay, pending.dir, x, y, true};
}

// ---------------------------------------------------------------- own body
// The simulator draws the whole body into the grid; the window shows at most a few segments. So
// the bot tracks it: this turn's visible chain, then the rest of last turn's body, shifted by
// however many steps the head moved -- worked out by matching the chain against last turn's body,
// not from the action, so a portal or a replayed transcript cannot desynchronise it.
std::deque<std::pair<int, int>> body;       // world cells, head first
// Per body index: a guess (track_body's straight-on continuation, or carried from one), not seen. The
// grid still draws guesses, but the mask never lets a guessed cell vacate: a coiled spawn splitting in
// place guessed her tail onto a visible segment of hers and the mask let her move into it (2026-10-09
// parity, loong_9002_0083 seed 1000, g013_s13_seg2's queen).
std::deque<bool> body_guess;
// Every cell the head has stood on, newest first: the body is always the newest `length` of them
// (a move adds cells at the head and drops them at the tail; a split keeps the head half). Unlike
// the guess below it is right after a portal. Used only while it agrees with every visible segment.
std::deque<std::pair<int, int>> trail;
// The cells last turn's MOVE puts the head on, step by step (plan_path), for when the window does not
// show the whole path: a far sprint whose middle runs out of it (loong_9002_0083's queen, 2026-10-09
// parity: the trail lost those cells and every index behind them was off). Empty when the path
// crosses a portal, which the bot cannot follow.
std::vector<std::pair<int, int>> planned;

void plan_path(obs::Snapshot const& s, int const* dirs, int n) {
    planned.clear();
    int x = s.hx, y = s.hy;
    for (int i = 0; i < n; i++) {
        int const c = s.cell_of(x, y);
        if (c < 0 || s.tile(c).edges[(std::size_t)dirs[i]].edge_type == unswbc::EdgeType::PORTAL) {
            planned.clear();
            return;
        }
        x = wrap(x + obs::DX[(std::size_t)dirs[i]], W_);
        y = wrap(y + obs::DY[(std::size_t)dirs[i]], H_);
        planned.emplace_back(x, y);
    }
}

// The tail, when the window shows it: an own segment that no own segment points at, whose other three
// sides (not the one it points through, to the segment ahead) are in the window and none a portal, so
// nothing could point at it unseen. Returned with the
// segments ahead of it, tail first, as far as the window shows them -- indices L-1, L-2, ... exactly.
// A spawn coiled out of the window and back (loong_9002_0097's queen) shows its tail beside its head,
// cut off from the head's chain; without this its indices were guessed and the mask let it move into
// its own body (2026-10-05 parity).
std::vector<std::pair<int, int>> tail_fragment(obs::Snapshot const& s) {
    std::array<bool, obs::CELLS> own{}, pointed{};
    auto own_seg = [&](int c) {
        auto const* part = s.tile(c).get_dragon();
        return part && part->dragon_id == my_id && !part->is_dragon_head;
    };
    auto ahead = [&](int c) {                      // the window cell the segment at c points at, or -1
        auto const& t = s.tile(c);
        int const d = obs::dir_index(t.get_dragon()->dir);
        if (t.edges[(std::size_t)d].edge_type != unswbc::EdgeType::EMPTY) return -1;
        return s.cell_of(wrap(t.position.x + obs::DX[(std::size_t)d], W_), wrap(t.position.y + obs::DY[(std::size_t)d], H_));
    };
    for (int c = 0; c < obs::CELLS; c++) own[(std::size_t)c] = own_seg(c);
    for (int c = 0; c < obs::CELLS; c++)
        if (own[(std::size_t)c]) {
            int const a = ahead(c);
            if (a >= 0) pointed[(std::size_t)a] = true;
        }
    int tail = -1;
    for (int c = 0; c < obs::CELLS; c++) {
        if (!own[(std::size_t)c] || pointed[(std::size_t)c]) continue;
        auto const& t = s.tile(c);
        int const fwd = obs::dir_index(t.get_dragon()->dir);   // where the segment ahead is: never behind
        bool closed = true;
        for (int d = 0; d < 4 && closed; d++) {
            if (d == fwd) continue;
            if (t.edges[(std::size_t)d].edge_type == unswbc::EdgeType::PORTAL) closed = false;
            else if (s.cell_of(wrap(t.position.x + obs::DX[(std::size_t)d], W_), wrap(t.position.y + obs::DY[(std::size_t)d], H_)) < 0)
                closed = false;
        }
        if (!closed) continue;
        if (tail >= 0) return {};                  // two candidates: not sure which, so no claim
        tail = c;
    }
    std::vector<std::pair<int, int>> frag;
    std::array<bool, obs::CELLS> seen{};
    for (int c = tail; c >= 0 && own[(std::size_t)c] && !seen[(std::size_t)c]; c = ahead(c)) {
        seen[(std::size_t)c] = true;
        frag.emplace_back(s.tile(c).position.x, s.tile(c).position.y);
    }
    return frag;
}

void track_body(obs::Snapshot const& s) {
    std::vector<std::pair<int, int>> vis;
    for (int i = 0; i < s.chain_len(); i++) {
        auto const& p = s.tile(s.chain_cell(i)).position;
        vis.emplace_back(p.x, p.y);
    }
    if (vis.empty()) vis.emplace_back(s.hx, s.hy);
    // the cells the head crossed since last turn: the visible chain up to where it meets the old
    // head (a sprint's middle cells), else just the new head (a single step, or a portal the chain
    // does not show)
    if (trail.empty() || trail.front() != std::make_pair(s.hx, s.hy)) {
        int join = -1;
        if (!trail.empty())
            for (int n = 1; n < (int)vis.size() && n <= obs::FAR_MAX_STEPS && join < 0; n++)
                if (vis[(std::size_t)n] == trail.front()) join = n;
        // no join in the window: last turn's planned path, if it ends on this head (a move cut short
        // ends on an earlier step) and agrees with every visible segment
        int k = -1;
        if (join < 0 && !trail.empty())
            for (int i = 0; i < (int)planned.size() && k < 0; i++)
                if (planned[(std::size_t)i] == std::make_pair(s.hx, s.hy)) k = i;
        bool use_plan = k >= 0;
        for (int i = 0; use_plan && i < (int)vis.size(); i++)
            use_plan = (i <= k ? planned[(std::size_t)(k - i)] : trail[(std::size_t)(i - k - 1)]) == vis[(std::size_t)i];
        if (use_plan)
            for (int i = 0; i <= k; i++) trail.push_front(planned[(std::size_t)i]);
        else
            for (int i = (join > 0 ? join : 1) - 1; i >= 0; i--) trail.push_front(vis[(std::size_t)i]);
    }
    planned.clear();
    while (trail.size() > 160) trail.pop_back();
    if ((int)trail.size() >= s.length) {
        bool ok = true;
        for (std::size_t i = 0; i < vis.size() && ok; i++) ok = trail[i] == vis[i];
        if (ok) {
            body.assign(trail.begin(), trail.begin() + s.length);
            body_guess.assign(body.size(), false);
            return;
        }
    }
    std::deque<std::pair<int, int>> nb(vis.begin(), vis.end());
    std::deque<bool> ng(vis.size(), false);
    int const v = (int)vis.size();
    if (!body.empty()) {
        // shift n: vis[n + j] == old[j] for the overlap
        int shift = -1;
        for (int n = 0; n <= v && shift < 0; n++) {
            bool ok = n < v;
            for (int j = 0; n + j < v && j < (int)body.size(); j++)
                if (vis[(std::size_t)(n + j)] != body[(std::size_t)j]) { ok = false; break; }
            if (ok) shift = n;
        }
        if (shift >= 0)
            for (std::size_t j = (std::size_t)(v - shift); j < body.size(); j++) {
                nb.push_back(body[j]);
                ng.push_back(j < body_guess.size() ? body_guess[j] : true);
            }
    }
    // a body we have never seen the end of (a dragon's first turns, a split child): continue
    // straight on from the last known link
    while ((int)nb.size() < s.length) {
        std::pair<int, int> const last = nb.back();
        int dx = 0, dy = 0;
        if (nb.size() >= 2) {
            auto const prev = nb[nb.size() - 2];
            dx = last.first - prev.first;
            dy = last.second - prev.second;
            if (dx > 1) dx = -1;
            if (dx < -1) dx = 1;
            if (dy > 1) dy = -1;
            if (dy < -1) dy = 1;
        } else {
            dx = -obs::DX[(std::size_t)s.facing];
            dy = -obs::DY[(std::size_t)s.facing];
        }
        nb.emplace_back(wrap(last.first + dx, W_), wrap(last.second + dy, H_));
        ng.push_back(true);
    }
    while ((int)nb.size() > s.length) nb.pop_back();
    while (ng.size() > nb.size()) ng.pop_back();
    // the tail's end of the body, where the window shows it (never over the head's own chain)
    auto const frag = tail_fragment(s);
    for (int i = 0; i < (int)frag.size(); i++) {
        int const idx = s.length - 1 - i;
        if (idx < v) break;
        nb[(std::size_t)idx] = frag[(std::size_t)i];
        ng[(std::size_t)idx] = false;
    }

    body = std::move(nb);
    body_guess = std::move(ng);
}

// ---------------------------------------------------------------- the grid
float loc[obs::N_CHANNELS * obs::CELLS];
float gf[gc::N];                             // the simulator's float grid, all 57 channels
alignas(16) i16 grid[NCH * GP];              // the CNN's input, Q12, planes padded to GP
float scal[N_SCAL + N_IDENT + N_AHIST + 1];  // what mlp0 reads after the LayerNorm (+ the zero pad)
// BC_AHIST (bc_vec.hpp VecEnv::note_action): the last action, and the decayed history of the ones
// before it -- a(t-2) 0.5, a(t-3) 0.25, ... per action id -- that ahist_in reads
int last_act = -1;
float ahist[N_AHIST] = {};

void render(obs::Snapshot const& s, int round, float units, float const* echo) {
    using namespace gc;
    std::memset(gf, 0, sizeof(gf));
    dm.grid(s.hx, s.hy, s.facing, round, gf);
    auto gch = [&](int c, int r, int col) -> float& { return gf[(std::size_t)c * CELLS + r * G + col]; };
    auto fill = [&](int c, float v) {
        float* pl = gf + (std::size_t)c * CELLS;
        for (int i = 0; i < CELLS; i++) pl[i] = v;
    };
    // live: the window sits at rows/cols HALF-VISION .. HALF+VISION
    static constexpr int LIVE[10][2] = {
        {obs::C_PEARL, PEARL}, {obs::C_PEARL_TIME, PEARL_TIMER},
        {obs::C_ALLY_HEAD, ALLY_HEAD}, {obs::C_ALLY_BODY, ALLY_BODY},
        {obs::C_ENEMY_HEAD, ENEMY_HEAD}, {obs::C_ENEMY_BODY, ENEMY_BODY},
        {obs::C_FACE, SEG_DIR}, {obs::C_FACE + 1, SEG_DIR + 1},
        {obs::C_FACE + 2, SEG_DIR + 2}, {obs::C_FACE + 3, SEG_DIR + 3}};
    for (int r = 0; r < obs::WINDOW; r++)
        for (int c = 0; c < obs::WINDOW; c++)
            for (auto const& lv : LIVE)
                gch(lv[1], r + HALF - obs::VISION, c + HALF - obs::VISION) = loc[lv[0] * obs::CELLS + r * obs::WINDOW + c];
    // self: the tracked body, head excluded, as bc_obs.hpp draws the true one
    int const len = (int)body.size();
    for (int i = 1; i < len; i++) {
        int ddx = body[(std::size_t)i].first - s.hx, ddy = body[(std::size_t)i].second - s.hy;
        torus(ddx, ddy);
        int ox, oy;
        world_to_ego(s.facing, ddx, ddy, ox, oy);
        if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
        gch(SELF_BODY, oy + HALF, ox + HALF) = 1.0f;
        gch(SELF_INDEX, oy + HALF, ox + HALF) = (float)i / (float)std::max(1, len - 1);
        if (i == len - 1) gch(SELF_TAIL, oy + HALF, ox + HALF) = 1.0f;
    }
    float const glob[10] = {(float)round / (float)MAX_ROUNDS, std::min(s.length, 64) / 64.0f,
                            units, (float)W_ / 64.0f, (float)H_ / 64.0f,
                            echo[0], echo[1], echo[2], echo[3], echo[4]};
    for (int k = 0; k < 10; k++) fill(ROUND + k, glob[k]);
    // each reporter at its head: its length (BC_PORTALREP: 39-42 carry the portal reports instead)
    static constexpr int UX[4] = {0, 1, 0, -1}, UY[4] = {-1, 0, 1, 0};
    for (auto const& p : reports) {
        int ddx = p.hx - s.hx, ddy = p.hy - s.hy;
        torus(ddx, ddy);
        int ox, oy;
        world_to_ego(s.facing, ddx, ddy, ox, oy);
        if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
        gch(REP_LEN, oy + HALF, ox + HALF) = std::min(p.len, 63) / 63.0f;
    }
    // what each known portal leads to, on the tile across its edge; where two land on one tile,
    // the larger value of each channel
    for (int i = 0; i < dm.n_portal; i++) {
        auto const& pr = dm.portal_rep[(std::size_t)i];
        int const bx = pr.tile % W_ + UX[pr.dir], by = pr.tile / W_ + UY[pr.dir];
        int ddx = bx - s.hx, ddy = by - s.hy;
        ddx = ((ddx % W_) + W_) % W_;
        ddy = ((ddy % H_) + H_) % H_;
        if (ddx > W_ / 2) ddx -= W_;
        if (ddy > H_ / 2) ddy -= H_;
        int ox, oy;
        world_to_ego(s.facing, ddx, ddy, ox, oy);
        if (ox < -HALF || ox >= G - HALF || oy < -HALF || oy >= G - HALF) continue;
        int const r = oy + HALF, c = ox + HALF;
        float const room = pr.box > 0 ? (float)pr.box / 3.0f : 0.0f;
        float const pearls = (float)pr.pearls / 3.0f * bc::GridDecay::at(bc::GRID_DECAY.pearl, round - pr.round);
        gch(PT_KNOWN, r, c) = 1.0f;
        gch(PT_CLOSED, r, c) = std::max(gch(PT_CLOSED, r, c), pr.box > 0 ? 1.0f : 0.0f);
        gch(PT_ROOM, r, c) = std::max(gch(PT_ROOM, r, c), room);
        gch(PT_PEARLS, r, c) = std::max(gch(PT_PEARLS, r, c), pearls);
    }
    // the queens
    if (my_id < 2) fill(IS_QUEEN, 1.0f);
    for (int r = 0; r < obs::WINDOW; r++)
        for (int c = 0; c < obs::WINDOW; c++) {
            gch(ALLY_QUEEN, r + HALF - obs::VISION, c + HALF - obs::VISION) = qwin[0][r * obs::WINDOW + c];
            gch(ENEMY_QUEEN, r + HALF - obs::VISION, c + HALF - obs::VISION) = qwin[1][r * obs::WINDOW + c];
        }
    static constexpr int Q_MEM[2] = {ALLY_QUEEN_MEM, ENEMY_QUEEN_MEM};
    static constexpr int Q_VEC[2] = {AQ_DX, EQ_DX};
    for (int side = 0; side < 2; side++) {
        auto const& q = dm.queen[side];
        if (q.round <= bc::mem_cfg::NEVER) continue;            // never known: all zero
        float const fresh = bc::GridDecay::at(bc::GRID_DECAY.seen, round - q.round);
        if (fresh <= 0.0f) continue;
        int ddx = q.x - s.hx, ddy = q.y - s.hy;
        torus(ddx, ddy);
        int ox, oy;
        world_to_ego(s.facing, ddx, ddy, ox, oy);
        fill(Q_VEC[side], std::max(-1.0f, std::min(1.0f, ox / 32.0f)));
        fill(Q_VEC[side] + 1, std::max(-1.0f, std::min(1.0f, oy / 32.0f)));
        fill(Q_VEC[side] + 2, fresh);
        if (ox >= -HALF && ox < G - HALF && oy >= -HALF && oy < G - HALF)
            gch(Q_MEM[side], oy + HALF, ox + HALF) = fresh;
    }
    // who this dragon is: birth round / 500, sin(id / 7), sin(id / 43)
    fill(BIRTH, (float)birth / 500.0f);
    fill(ID7, std::sin((float)my_id / 7.0f));
    fill(ID43, std::sin((float)my_id / 43.0f));

    // the CNN's channels as Q12, and mlp0's scalars off the constant planes (ff_net.py scal_feats,
    // ident_feats): round, round^2, length, units^2, is_queen; birth, sin(id/7), sin(id/43)
    for (int c = 0; c < NCH; c++) {
        float const* src = gf + (std::size_t)c * CELLS;
        i16* dst = grid + (std::size_t)c * GP;
        for (int i = 0; i < CELLS; i++) dst[i] = ln::to_q(src[i], ln::QG);
        for (int i = CELLS; i < GP; i++) dst[i] = 0;
    }
    float const r = gf[ROUND * CELLS + CENTRE], l = gf[LENGTH * CELLS + CENTRE];
    float const u = gf[UNITS * CELLS + CENTRE], q = gf[IS_QUEEN * CELLS + CENTRE];
    float const sc[N_SCAL] = {r, r * r, l, u * u, q};
    int k = 0;
    if (wt::SCAL_IN)
        for (float v : sc) scal[k++] = v;
    if (wt::IDENT_IN)
        for (int c = BIRTH; c < BIRTH + N_IDENT; c++) scal[k++] = gf[c * CELLS + CENTRE];
    if (wt::AHIST_IN)
        for (int a = 0; a < N_AHIST; a++) scal[k++] = ahist[a];
    while (k < MLP_IN - XIN) scal[k++] = 0.0f;
}

// ---------------------------------------------------------------- forward
constexpr int MAXC = wt::C1 > wt::C2 ? wt::C1 : wt::C2;
constexpr int MAXIN = NCH > MAXC ? NCH : MAXC;
alignas(16) i16 act_a[MAXC * GP], act_b[MAXC * GP], act_c[MAXC * GP];
// im2col_s1 writes up to 7 cells past a row; slack so the last pair cannot run off `col`
alignas(16) i16 col[MAXIN * 9 * GP + 64];
alignas(16) i16 padbuf[(G + 2) * (G + 2)];
alignas(16) i16 pads[MAXIN * (G + 2) * (G + 2) + 32];
alignas(16) i32 acc[MAXC * GP];
float zbuf[MAXC * GP];
alignas(16) i16 flat[wt::SQ * DC + 8];
alignas(16) i16 xin[MLP_IN + 8], hq[wt::HID + 8];
float xf[XIN], zh[wt::HID];
int prev_action = NO_ACTION;

#ifdef BC_METER
std::int64_t m_gemm = 0, m_im2col = 0, m_post = 0;
#endif
void conv3(Layer const& L, i16 const* in, float qin, int h, int w, int Pin, int stride, int ho, int wo,
           int Pout, i16 const* skip, i16* out) {
#ifdef BC_METER
    std::int64_t const a0 = points_now();
#endif
#if LN_SIMD
    if (stride == 1) ln::im2col_s1(in, L.cin, h, w, Pin, Pout, pads, col);
    else ln::im2col(in, L.cin, h, w, Pin, stride, ho, wo, Pout, padbuf, col);
#else
    ln::im2col(in, L.cin, h, w, Pin, stride, ho, wo, Pout, padbuf, col);
#endif
#ifdef BC_METER
    std::int64_t const a1 = points_now();
#endif
    ln::gemm(L.w, L.cout, L.cin * 9, col, Pout, acc);
#ifdef BC_METER
    std::int64_t const a2 = points_now();
#endif
    ln::dequant(acc, L.s, L.b, L.cout, Pout, ho * wo, qin, zbuf);
    ln::gn_silu(zbuf, L.cout, Pout, ho * wo, L.g, L.beta, skip, out);
#ifdef BC_METER
    m_im2col += a1 - a0; m_gemm += a2 - a1; m_post += points_now() - a2;
#endif
}

// a dense layer + SiLU on a Q10 row, into a Q10 row
void dense_silu(Layer const& L, i16 const* x, int n_in, i16* y) {
    ln::gemm(x, 1, n_in, L.w, L.cout, acc);
    ln::dequant_vec(acc, L.s, L.b, L.cout, zh);
    for (int i = 0; i < L.cout; i++) y[i] = ln::to_q(ln::silu(zh[i]), ln::QA);
}

void forward(float* logits) {
    conv3(stem, grid, ln::QG, G, G, GP, 1, G, G, GP, nullptr, act_a);
    for (int i = 0; i < wt::B1; i++) {
        conv3(res15[(std::size_t)(2 * i)], act_a, ln::QA, G, G, GP, 1, G, G, GP, nullptr, act_b);
        conv3(res15[(std::size_t)(2 * i + 1)], act_b, ln::QA, G, G, GP, 1, G, G, GP, act_a, act_c);
        std::memcpy(act_a, act_c, sizeof(i16) * wt::C1 * GP);
    }
    conv3(down, act_a, ln::QA, G, G, GP, 2, D, D, DC, nullptr, act_b);
    for (int i = 0; i < wt::B2; i++) {
        conv3(res8[(std::size_t)(2 * i)], act_b, ln::QA, D, D, DC, 1, D, D, DC, nullptr, act_c);
        conv3(res8[(std::size_t)(2 * i + 1)], act_c, ln::QA, D, D, DC, 1, D, D, DC, act_b, act_a);
        std::memcpy(act_b, act_a, sizeof(i16) * wt::C2 * DC);
    }
    // squeeze 1x1 -> SiLU -> flatten
    ln::interleave(act_b, wt::C2, DC, col);
    ln::gemm(squeeze.w, wt::SQ, wt::C2, col, DC, acc);
    ln::dequant(acc, squeeze.s, squeeze.b, wt::SQ, DC, DC, ln::QA, zbuf);
    for (int c = 0; c < wt::SQ; c++)
        for (int j = 0; j < DC; j++) flat[c * DC + j] = ln::to_q(ln::silu(zbuf[c * DC + j]), ln::QA);
    // embed -> SiLU, the previous action, LayerNorm
    ln::gemm(flat, 1, wt::SQ * DC, embed.w, wt::EMB, acc);
    ln::dequant_vec(acc, embed.s, embed.b, wt::EMB, zbuf);
    for (int i = 0; i < wt::EMB; i++) xf[i] = ln::silu(zbuf[i]);
    for (int i = 0; i < PREV_DIM; i++) xf[wt::EMB + i] = prev_table[prev_action * PREV_DIM + i];
    ln::layer_norm(xf, XIN, ln_g, ln_b);
    // mlp0 reads [LayerNorm, scalars, identity]; then the dense layers, then the policy
    for (int i = 0; i < XIN; i++) xin[i] = ln::to_q(xf[i], ln::QA);
    for (int i = XIN; i < MLP_IN; i++) xin[i] = ln::to_q(scal[i - XIN], ln::QA);
    dense_silu(mlp[0], xin, MLP_IN, hq);
    for (int l = 1; l < wt::LAYERS; l++) {
        dense_silu(mlp[(std::size_t)l], hq, wt::HID, xin);
        std::memcpy(hq, xin, sizeof(i16) * wt::HID);
    }
    ln::gemm(hq, 1, wt::HID, pi.w, PI_P, acc);
    ln::dequant_vec(acc, pi.s, pi.b, N_ACT, logits);
}

// ---------------------------------------------------------------- the reply
std::string out;

// The world directions the action walks (moves and far sprints), or 0 for a split / the self-kill.
int path_of(int action, obs::Snapshot const& snap, int* dirs) {
    if (action < obs::N_MOVES) {
        std::array<int, 3> d3{};
        int const n = obs::decode_move(action, snap.facing, d3);
        for (int i = 0; i < n; i++) dirs[i] = d3[(std::size_t)i];
        return n;
    }
    if (action >= obs::FAR_ID && action < obs::FAR_ID + obs::N_FAR) {
        int const k = action - obs::FAR_ID, n = snap.far_n[(std::size_t)k];
        for (int i = 0; i < n; i++) dirs[i] = snap.far_dirs[(std::size_t)(k * obs::FAR_MAX_STEPS + i)];
        return n;
    }
    return 0;
}

void emit(int action, obs::Snapshot const& snap, int const* dirs, int n) {
    if (action == obs::SUICIDE_ID) return;            // no MOVE and no SPLIT: the engine kills it
    if (n > 0) {
        static constexpr char NAMES[4] = {'N', 'E', 'S', 'W'};
        out += "MOVE ";
        for (int i = 0; i < n; i++) out += NAMES[dirs[i]];
        out += '\n';
        return;
    }
    int k;
    if (action >= obs::XSPLIT_ID && action < obs::XSPLIT_ID + obs::N_XSPLITS)
        k = snap.length - obs::XSPLIT_KEEP[(std::size_t)(action - obs::XSPLIT_ID)];
    else {
        k = obs::SPLIT_K[(std::size_t)(action - obs::N_MOVES)];
        if (k < 0) k = snap.length / 2;
    }
    out += "SPLIT ";
    out += std::to_string(k);
    out += '\n';
}

// The v2 packet four ways with protocol 3, as the simulator casts it; nothing on the self-kill
// (the dragon is dead).
void flush_turn(int action, std::uint64_t payload) {
#ifdef BC_DUMP
    out += "ACT " + std::to_string(action) + "\n";
#endif
    if (action != obs::SUICIDE_ID) {
        std::string const v = std::to_string((unsigned long long)payload);
        for (char d : {'N', 'E', 'S', 'W'}) {
            out += "SONAR ";
            out += d;
            out += ' ';
            out += v;
            out += '\n';
        }
        out += "PROTOCOL 3\n";
    }
    out += "ENDTURN\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
    out.clear();
}

}  // namespace

int main() {
    static char stdout_buf[1 << 12];
    std::setvbuf(stdout, stdout_buf, _IOFBF, sizeof(stdout_buf));
#ifdef BC_METER
    std::int64_t const t_start = points_now();
#endif
    auto [ct, game] = unswbc::init();
    bool const ok = load();
#ifdef BC_METER
    std::fprintf(stderr, "METER startup %lld\n", (long long)(points_now() - t_start));
#endif
#ifdef BC_DUMP
    std::FILE* dump = stderr;
    if (char const* f = std::getenv("BC_DUMP_FILE"))
        if (std::FILE* fh = std::fopen(f, "w")) dump = fh;      // wasm has no files: stderr (wasmrun.py)
#endif
    W_ = game.width;
    H_ = game.height;
    my_id = ct.get_id();
    dm.reset(W_, H_);

    obs::Snapshot snap;
    std::uint8_t mask[N_ACT];
    float logits[N_ACT];
    int dirs[obs::FAR_MAX_STEPS];
    while (unswbc::update(ct, game)) {
#ifdef BC_METER
        std::int64_t const t0 = points_now();
#endif
        int const round = game.round_num;
        // a map spawn's first turn is round 0, a split child's the round it was born in
        // (bc_core.hpp Dragon::birth)
        if (birth < 0) birth = round;
        snap.build(ct, game);
        snap.local(loc);
        float echo[obs::N_ECHO];
        obs::echo_scalars(ct, echo);
        land_trip(snap);
        observe(snap, round);
        int const team = snap.my_team == 'B' ? 1 : 0;
        hear(ct, snap, team, round);
        track_body(snap);
        {
            int cells[128];
            int const n = (int)body.size() < 128 ? (int)body.size() : 128;
            // a guessed index is no cell the mask knows (never vacates)
            for (int i = 0; i < n; i++)
                cells[i] = (std::size_t)i < body_guess.size() && body_guess[(std::size_t)i]
                    ? -1 : snap.cell_of(body[(std::size_t)i].first, body[(std::size_t)i].second);
            snap.set_body(cells, n);
        }
        // the queen guard: the policy trained with BC_QUEEN_GUARD=1 (both teams, gates included)
        snap.mask(ct, game, my_id < 2, mask);
        // the queen dead-end mask (user, 2026-10-09; the simulator's BC_QUEEN_DEADEND=BC_QD_LEVEL): never a
        // move into a dead end she can see while another move is not one; level 2 also counts her own body
        // as the dead end's wall and, when every move is fatal, keeps only the splits that let her out
        if (my_id < 2) snap.queen_deadend(mask, BC_QD_LEVEL);
        render(snap, round, (float)ct.get_unit_count() / (float)game.unit_limit, echo);
#ifdef BC_METER
        std::int64_t const t1 = points_now();
#endif
        int action = -1;
        if (ok) {
            forward(logits);
#ifdef BC_METER
            std::fprintf(stderr, "METER obs+render %lld forward %lld (convs: im2col %lld gemm %lld dequant+gn+silu %lld)\n",
                         (long long)(t1 - t0), (long long)(points_now() - t1), (long long)m_im2col,
                         (long long)m_gemm, (long long)m_post);
            m_im2col = m_gemm = m_post = 0;
#endif
            float best = -3.4e38f;
            for (int a = 0; a < N_ACT; a++)
                if (mask[a] && logits[a] > best) { best = logits[a]; action = a; }
        }
#ifdef BC_DUMP
        // parity testing only (wasmprobe/parity_ffl.py PARITY_FAR_TEST): play the best legal far sprint
        // whenever there is one, so their paths get checked against the simulator's
        static bool const far_test = std::getenv("BC_FAR_TEST") != nullptr;
        if (far_test) {
            float best = -3.4e38f;
            for (int a = obs::FAR_ID; a < obs::FAR_ID + obs::N_FAR; a++)
                if (mask[a] && logits[a] > best) { best = logits[a]; action = a; }
        }
#endif
        if (action < 0)                                       // no net, or no legal move
            for (int a = 0; a < N_ACT && action < 0; a++)
                if (mask[a]) action = a;
        if (action < 0) action = obs::SUICIDE_ID;
        // the packet describes this dragon before its move
        std::uint64_t const payload = packet(snap, team, round);
#ifdef BC_DUMP
        std::fprintf(dump, "DUMP %d", prev_action);
        for (int c = 0; c < NCH; c++)
            for (int i = 0; i < GC; i++) std::fprintf(dump, " %d", (int)grid[c * GP + i]);
        for (int i = 0; i < MLP_IN - XIN; i++) std::fprintf(dump, " %.9g", scal[i]);
        for (int a = 0; a < N_ACT; a++) std::fprintf(dump, " %d", (int)mask[a]);
        for (int a = 0; a < N_ACT; a++) std::fprintf(dump, " %.6g", ok ? logits[a] : 0.0f);
        std::fprintf(dump, " %d %llu\n", action, (unsigned long long)payload);
#if !LN_SIMD
        std::fprintf(dump, "ACCPEAK %lld\n", (long long)ln::acc_peak);
#endif
        std::fflush(dump);
#endif
        out += ok ? "INDICATOR ffl\n" : "INDICATOR NO NET\n";
        int const n = path_of(action, snap, dirs);
        if (n > 0) plan_trip(snap, dirs, n);
        else pending.active = false;
        plan_path(snap, dirs, n);
        prev_action = action;
        // VecEnv::note_action, as the simulator steps this action
        for (float& v : ahist) v *= 0.5f;
        if (last_act >= 0) ahist[last_act] += 0.5f;
        last_act = action;
        emit(action, snap, dirs, n);
        last_sent = payload;
        sent_any = true;
        flush_turn(action, payload);
    }
}
