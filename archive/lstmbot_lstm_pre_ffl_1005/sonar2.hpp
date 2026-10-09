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
//   [37:32] enemy x  the nearest enemy head in the sender's window (Chebyshev, torus)
//   [31:26] enemy y
//   [25:20] P(split)            6 bits, x/63
//   [19:14] P(sprint)           6 bits: 2- and 3-step moves
//   [13:12] facing              the sender's, absolute (0 N, 1 E, 2 S, 3 W)
//   [11:7]  P(first step left)  5 bits, x/31, relative to that facing
//   [6:2]   P(first step right)
//   [1]     enemy valid
//   [0]     zero
//
// Everything describes the sender as it writes its reply, BEFORE its move (the engine
// casts the ray after the move, but a bot cannot know where the move will leave it):
// its head, length, facing and window then, and the probabilities of the move it is
// making. "Straight" is what is left: 1 - split - left - right. A receiver drops a
// packet equal to the last one it sent itself (its own ray coming back).
#pragma once

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
    bool enemy = false;
    int ex = 0, ey = 0;
    float p_split = 0, p_sprint = 0, p_left = 0, p_right = 0;
    int facing = 0;
};

// The bits a sender fills from its own state: head, length, enemy, facing and the
// enemy flag -- not the probabilities, not the tag. A dragon recognises its own
// packet coming back by these (its probabilities may be recomputed differently).
constexpr uint64_t STATE_BITS = ((1ull << 56) - 1) & ~(((1ull << 12) - 1) << 14) & ~(((1ull << 10) - 1) << 2);
inline bool same_sender(uint64_t a, uint64_t b) { return ((a ^ b) & STATE_BITS) == 0; }

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
    }
    v |= q(p.p_split, 63) << 20;
    v |= q(p.p_sprint, 63) << 14;
    v |= (uint64_t)(p.facing & 3) << 12;
    v |= q(p.p_left, 31) << 7;
    v |= q(p.p_right, 31) << 2;
    return v | ((uint64_t)tag(team, round, v) << 56);
}

// true and the packet (and the round it was sent in) when the tag checks for
// `team` at `round` or `round - 1`.
inline bool decode(uint64_t v, int team, int round, Packet& p, int& sent_round) {
    const uint64_t low = v & 0x00FFFFFFFFFFFFFFull;
    const uint8_t t = (uint8_t)(v >> 56);
    if ((low & 1ull) != 0) return false;
    sent_round = -1;
    for (int r = round; r >= std::max(0, round - 1); r--)
        if (tag(team, r, low) == t) { sent_round = r; break; }
    if (sent_round < 0) return false;
    p.hx = (int)(low >> 50 & 63);
    p.hy = (int)(low >> 44 & 63);
    p.len = (int)(low >> 38 & 63);
    p.enemy = (low & 2ull) != 0;
    p.ex = (int)(low >> 32 & 63);
    p.ey = (int)(low >> 26 & 63);
    p.p_split = (float)(low >> 20 & 63) / 63.0f;
    p.p_sprint = (float)(low >> 14 & 63) / 63.0f;
    p.facing = (int)(low >> 12 & 3);
    p.p_left = (float)(low >> 7 & 31) / 31.0f;
    p.p_right = (float)(low >> 2 & 31) / 31.0f;
    return true;
}

}  // namespace s2
}  // namespace bc
