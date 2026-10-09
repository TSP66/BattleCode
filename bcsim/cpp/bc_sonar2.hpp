// Sonar v2: the team packet (user's design, 2026-09-29/30).
//
// Every dragon of a team that speaks it casts the same 64 bits in all four
// directions after its move. Nothing here reads anything a deployed bot cannot
// know: its own head, length and facing, the enemy heads in its own 7x7 window,
// and the probabilities its own policy gave this turn's action.
//
//   [63:56] tag      8 bits: a keyed hash of (team, round, bits 55..0). A receiver
//                    accepts a packet only if the tag checks for its own team at
//                    the current round or the one before (an inbox spans the end of
//                    one round and the start of the next). Rejects the other team's
//                    packets, stale replays and random payloads (1 in 256 each). A
//                    same-round verbatim copy by the enemy still passes, and does no
//                    harm: every field is an absolute fact about one of our dragons.
//   [55:50] head x   absolute, 6 bits (maps are at most 64 wide)
//   [49:44] head y
//   [43:38] length   min(len, 63)
//   [37:32] enemy x  the enemy QUEEN if any of it is in the sender's window (its head
//   [31:26] enemy y  if visible, else its nearest segment), else the nearest enemy head
//                    (Chebyshev, torus)
//   [25:21] P(split)            5 bits, x/31
//   [20]    enemy is the queen  (2026-10-01)
//   [19:16] P(sprint)           4 bits, x/15: 2- and 3-step moves
//   [15:14] queen age           rounds since the enemy queen was at (ex, ey): 0 = in the sender's
//                               window now, 1-3 = RELAYED from its memory (its own sighting or a
//                               teammate's report), so the news crosses the team hop by hop
//   [13:12] facing              the sender's, absolute (0 N, 1 E, 2 S, 3 W)
//   [11:8]  P(first step left)  4 bits, x/15, relative to that facing
//   [7:4]   P(first step right)
//   [3:2]   came from           a relayed queen report: which way (absolute, from the sender) the
//                               news came -- towards the teammate it heard it from, or towards the
//                               queen for its own sighting. A receiver lying that way ignores it:
//                               news is never relayed back to where it came from
//   [1]     enemy valid
//   [0]     the sender is our queen (2026-10-01; was a zero bit, which the tag makes
//           redundant as a filter)
//
// BC_PORTALREP (2026-10-02, user): the four probabilities and the facing were no use, so
// those 19 bits carry instead the PORTAL REPORT -- what lies on the far side of the last
// portal the sender went through. Every other field keeps its place.
//
//   [13]    portal valid    the sender has been through a portal (and lived)
//   [11:10] portal dir      the absolute direction it stepped through it (0 N, 1 E, 2 S, 3 W)
//   [9:4]   portal x        the ENTRANCE: the tile it stepped from, absolute. With the direction
//   [12],[25:21] portal y   this names one side of one portal edge exactly (y: bit 5 at [12],
//                           bits 4..0 at [25:21]); the receiver needs nothing else to find it
//   [17:16] box             where it came out: 0 = not known to be closed; 1-3 = a region closed
//                           by kelp and portal edges, as the sender's memory proves it, of
//                           <= 4 / <= 25 / <= 100 tiles (portal_box_bucket)
//   [19:18] pearls          pearls expected there, 0..3 (3 = 3 or more): the whole box when
//                           closed, else within 4 steps of the exit tile
//
// Everything describes the sender as it writes its reply, BEFORE its move (the engine
// casts the ray after the move, but a bot cannot know where the move will leave it):
// its head, length, facing and window then, and the probabilities of the move it is
// making. "Straight" is what is left: 1 - split - left - right. A receiver drops a
// packet equal to the last one it sent itself (its own ray coming back).
#pragma once

#if defined(BC_PORTALREP) && !defined(BC_SONAR2)
#error "BC_PORTALREP is a sonar v2 packet layout: build it with BC_SONAR2"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bc {
namespace s2 {

constexpr uint64_t KEY = 0xC2B2AE3D27D4EB4Full;

inline uint64_t mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline uint8_t tag(int team, int round, uint64_t low56) {
    const uint64_t k = KEY ^ ((uint64_t)(team & 1) << 63) ^ ((uint64_t)(uint32_t)round << 20);
    return (uint8_t)(mix(k ^ mix(low56 & 0x00FFFFFFFFFFFFFFull)) >> 56);
}

struct Packet {
    int hx = 0, hy = 0, len = 0;
    bool queen = false;          // the sender is its team's queen
    bool enemy = false;
    bool enemy_queen = false;    // (ex, ey) is the enemy queen
    int queen_age = 0;           // rounds since it was there (0..RELAY_ROUNDS)
    int came_from = 0;           // a relayed report: absolute direction it came from (0 N, 1 E, 2 S, 3 W)
    int ex = 0, ey = 0;
    float p_split = 0, p_sprint = 0, p_left = 0, p_right = 0;
    int facing = 0;
    // BC_PORTALREP: the last portal the sender went through (see the layout above)
    bool portal = false;
    int px = 0, py = 0, pdir = 0;   // the entrance tile and the direction stepped
    int pbox = 0, ppearls = 0;      // what is on the far side
};

// BC_PORTALREP: the far side's region size -> the 2-bit box field (0 = not closed, or too big)
constexpr int PORTAL_BOX_MAX = 100;
constexpr int PORTAL_PEARL_STEPS = 4;   // an open far side: pearls within this many steps
inline int portal_box_bucket(int tiles) {
    if (tiles <= 0 || tiles > PORTAL_BOX_MAX) return 0;
    return tiles <= 4 ? 1 : tiles <= 25 ? 2 : 3;
}

// The bits a sender fills from its own state: head, length, enemy, facing and the
// flags -- not the probabilities, not the tag. A dragon recognises its own packet
// coming back by these (its probabilities may be recomputed differently).
#ifdef BC_PORTALREP
constexpr uint64_t PROB_BITS = 0;    // every field is the sender's state
#else
constexpr uint64_t PROB_BITS = (((1ull << 5) - 1) << 21) | (((1ull << 4) - 1) << 16) | (((1ull << 8) - 1) << 4);
#endif
constexpr int RELAY_ROUNDS = 3;      // an enemy queen position older than this is not passed on
constexpr uint64_t STATE_BITS = ((1ull << 56) - 1) & ~PROB_BITS;
inline bool same_sender(uint64_t a, uint64_t b) { return ((a ^ b) & STATE_BITS) == 0; }

// The absolute direction (0 N, 1 E, 2 S, 3 W) of an offset, by its larger axis (ties: vertical).
inline int bearing(int dx, int dy) {
    if (std::abs(dy) >= std::abs(dx)) return dy < 0 ? 0 : 2;
    return dx > 0 ? 1 : 3;
}

inline uint64_t q(float p, int levels) {
    const float c = std::min(1.0f, std::max(0.0f, p));
    return (uint64_t)std::lround(c * (float)levels);
}

inline uint64_t encode(const Packet& p, int team, int round) {
    uint64_t v = 0;
    v |= (uint64_t)(p.hx & 63) << 50;
    v |= (uint64_t)(p.hy & 63) << 44;
    v |= (uint64_t)std::min(p.len, 63) << 38;
    if (p.enemy) {
        v |= (uint64_t)(p.ex & 63) << 32;
        v |= (uint64_t)(p.ey & 63) << 26;
        v |= 2ull;
        if (p.enemy_queen) {
            v |= 1ull << 20;
            v |= (uint64_t)std::min(std::max(p.queen_age, 0), 3) << 14;
            v |= (uint64_t)(p.came_from & 3) << 2;
        }
    }
    if (p.queen) v |= 1ull;
#ifdef BC_PORTALREP
    if (p.portal) {
        v |= 1ull << 13;
        v |= (uint64_t)(p.pdir & 3) << 10;
        v |= (uint64_t)(p.px & 63) << 4;
        v |= (uint64_t)(p.py >> 5 & 1) << 12;
        v |= (uint64_t)(p.py & 31) << 21;
        v |= (uint64_t)(p.pbox & 3) << 16;
        v |= (uint64_t)std::min(std::max(p.ppearls, 0), 3) << 18;
    }
#else
    v |= q(p.p_split, 31) << 21;
    v |= q(p.p_sprint, 15) << 16;
    v |= (uint64_t)(p.facing & 3) << 12;
    v |= q(p.p_left, 15) << 8;
    v |= q(p.p_right, 15) << 4;
#endif
    return v | ((uint64_t)tag(team, round, v) << 56);
}

// true and the packet (and the round it was sent in) when the tag checks for
// `team` at `round` or `round - 1`.
inline bool decode(uint64_t v, int team, int round, Packet& p, int& sent_round) {
    const uint64_t low = v & 0x00FFFFFFFFFFFFFFull;
    const uint8_t t = (uint8_t)(v >> 56);
    sent_round = -1;
    for (int r = round; r >= std::max(0, round - 1); r--)
        if (tag(team, r, low) == t) { sent_round = r; break; }
    if (sent_round < 0) return false;
    p.hx = (int)(low >> 50 & 63);
    p.hy = (int)(low >> 44 & 63);
    p.len = (int)(low >> 38 & 63);
    p.queen = (low & 1ull) != 0;
    p.enemy = (low & 2ull) != 0;
    p.enemy_queen = p.enemy && (low >> 20 & 1ull) != 0;
    p.ex = (int)(low >> 32 & 63);
    p.ey = (int)(low >> 26 & 63);
    p.queen_age = p.enemy_queen ? (int)(low >> 14 & 3) : 0;
    p.came_from = p.enemy_queen ? (int)(low >> 2 & 3) : 0;
#ifdef BC_PORTALREP
    p.portal = (low >> 13 & 1ull) != 0;
    if (p.portal) {
        p.pdir = (int)(low >> 10 & 3);
        p.px = (int)(low >> 4 & 63);
        p.py = (int)((low >> 12 & 1) << 5 | (low >> 21 & 31));
        p.pbox = (int)(low >> 16 & 3);
        p.ppearls = (int)(low >> 18 & 3);
    }
#else
    p.p_split = (float)(low >> 21 & 31) / 31.0f;
    p.p_sprint = (float)(low >> 16 & 15) / 15.0f;
    p.facing = (int)(low >> 12 & 3);
    p.p_left = (float)(low >> 8 & 15) / 15.0f;
    p.p_right = (float)(low >> 4 & 15) / 15.0f;
#endif
    return true;
}

}  // namespace s2
}  // namespace bc
