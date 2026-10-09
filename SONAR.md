# Sonar — SOLVED, and byte-identical (unswbc 1.2.3)

**Status (2026-09-24): our simulator matches the reference engine exactly.**
`tests/parity_sonar.py` drives the engine and our text simulator in lockstep and
compares every round block byte for byte, across three protocol regimes on all
ten official maps: **0 blocks differing out of 423,669 turns, carrying 1,159,721
messages.** The legacy path is unchanged and still exact.

    python bcsim/tests/parity_sonar.py       # lockstep block parity, 3 regimes
    python bcsim/tests/stress.py 300         # every rule, random maps and policies

**2026-10-01:** the drag rule below was refined against unswbc 1.2.3 (stress game gen866):
the trigger is the CAST DIRECTION being opposite the dragon's facing, not the first
step entering the neck. They differ only where a portal put the neck elsewhere. The
one-off probes that derived these rules (probe_sonar*.py, sonar_truth.py) were deleted
then; they targeted 1.0.0 and encoded the old trigger.

## What unlocked it: the engine's replay is the source of truth

Every earlier attempt inferred the rules from *inside a bot*, where the
observation block is deliberately lossy — `ECHOES` is one aggregate with no
bearing and messages carry no sender, so a targeting bug and a classification bug
look identical. Eight hypotheses were tested and rejected that way.

The engine ships a Cap'n Proto replay, and it records **every single ray**:

    EventSonarPing { senderId, direction, value64, origin, end, hitId, hitKind }

`origin` and `end` are the tiles the ray started and stopped on, `hitId` the
dragon it reached, `hitKind` one of `empty/kelp/ally/ally_head/enemy/enemy_head`.
That is per-ray, tile-level ground truth — no inference at all.

`tests/replay.py` reads it (packed encoding, far pointers, the lot). There is no
shipped `.capnp` schema; the struct layout was read off the accessors in the
replay viewer unswbc ships (`replay-viewer.vsix` →
`extension/dist/webview/webview.js`), which is generated from the real schema.
Struct ids and `formatVersion` are checked on every parse so a format change
fails loudly instead of being misread.

**Lesson worth keeping: when a black box ships a debug artefact, read it before
forming hypotheses about the black box.** This cost several sessions of guessing
that a 20-minute look at the `.vsix` would have prevented.

## The rule we had wrong: a ray is dragged along its own body

    A ray cast OPPOSITE TO THE DRAGON'S FACING is dragged the whole length of the
    body and re-emerges FROM THE TAIL, travelling along the last body link — not in
    the direction it was cast. (Until 2026-10-01: "if the first step enters the
    segment behind the head", which is the same thing except next to a portal.)

So a dragon curled into an L can cast **west** and have the ray leave going
**south**. Entering any *deeper* own segment is an ordinary hit on yourself,
which is how a curled dragon comes to hear its own sonar.

The old model cast a straight line and treated the whole body as transparent.
That agrees only when the body happens to lie straight behind the head, which is
why it looked nearly right: 97.2% per-ray on an empty torus, and 54.9% on `help`.

`tests/sonar_truth.py` (deleted 2026-10-01) reconstructed the board from the replay, predicted each ray
independently of our simulator, and compared: **176,704 rays, 100.00%, all ten
official maps, 0 state-rebuild mismatches.** The standing guard is now parity_sonar.py
and stress.py, which compare whole blocks against the engine.

Worth noting how the rule was pinned: dragon 1, body `[(15,11),(14,11),(14,12)]`,
cast **W**, and the engine logged the ray leaving **S** from `(14,12)`. Its four
rays came back as N, E, S, S — no W at all. A straight-line model cannot produce
that; only the body-following one can.

## The three rules about the protocol

Measured, each with a dedicated experiment, and all three are load-bearing:

1. **The protocol is per DRAGON, not per team.** When only dragon 0 declared
   protocol 3, its own teammate got no `ECHOES` line at all and every 64-bit
   message to it was dropped.
2. **A split child inherits its parent's protocol**, so it is not legacy at
   birth. A child split off in round 0 both received 64-bit payloads and carried
   an all-zero `ECHOES` line on its very first turn. The declaration therefore
   has to be applied *before* the action, or a child born this turn inherits a
   stale value — that was the last of the ten maps to go green.
3. **A payload wider than 32 bits is DROPPED for a receiver still on the legacy
   protocol**, not truncated. With one team declaring protocol 3 and the other
   not, 114 of 114 wide messages to a legacy dragon were dropped, while the same
   rays carrying a 32-bit payload all arrived. The sender's echo still counts the
   hit — only the message is withheld.

And one that is *not* a rule: **the directed `SONAR <dir> <u64>` form is accepted
whatever protocol the dragon has declared.** A dragon that never declares
protocol 3 still casts directed rays, and protocol-3 dragons still hear them. The
protocol governs only what a dragon *receives* (its `ECHOES` line, and whether a
wide payload can land). Gating the cast on protocol 3 — and gating the
body-dragging geometry on it — were both bugs, caught only by the mixed regime.

The mixed regime exists in `parity_sonar.py` precisely because neither of those
was visible when every dragon ran the same protocol.

## Message delivery timing

    a ray is resolved and delivered IMMEDIATELY, and the receiver reads it on its
    own next turn.

So a receiver later in the sending round reads it that same round (lag 0, 262
cases), one that has already acted reads it next round (lag +1, 227), and a
dragon that hits itself reads it next round (lag +1, 18). Inboxes are emptied at
the round boundary. This was already right and was never the bug.

Measured with round-unique payloads. An earlier pass of this measurement used a
payload that repeated every round, which made the pairing ambiguous and the lag
histogram meaningless — the same mistake as quoting aggregate message totals
below. **Tag every probe payload uniquely.**

## Wire format

### `PROTOCOL 3` gates everything

The engine speaks the new sonar only to a bot that asks, and the shipped helper
asks **on every turn**, immediately before `ENDTURN`:

    PROTOCOL 3
    ENDTURN

Not a one-time handshake. Without it: no `ECHOES` line, and `SONAR <uint64>`
capped at 32 bits.

**`mybot/` now speaks protocol 3, but only when the embedded net was trained with
sonar** (`embedded::SCALARS == 713`). A 14- or 708-scalar checkpoint keeps the
exact legacy behaviour — no `PROTOCOL`, no `SONAR`, no `ECHOES` in the blocks it
is sent. That gate matters: our own rays land on our own dragons, so broadcasting
makes `num_msgs` (scalar 12) non-zero, and a net trained without sonar saw that
column as always zero. Verified by emitting against a transcript: the 713 net
prints 4 `SONAR` lines plus `PROTOCOL 3` every turn, the 708 net prints neither.

### Sending

    SONAR <N|E|S|W> <uint64>     directed, full 64 bits
    SONAR <uint32>               legacy, along the current facing only

A dragon may send **one message per cardinal direction in the same turn** — four
`SONAR` lines, each with its own payload. Sonar does not consume the turn's
action; it is sent alongside `MOVE` or `SPLIT`, and is cast **after** the action,
from where the dragon ends its turn. The full 64 bits survive, top bit included.

### Receiving messages

    NUM_MSGS <n>
    <uint64>          x n

No sender id and no bearing — which is exactly why a child needs a tag byte to
recognise its parent. Several can land in one turn; four is reachable.

**The enemy receives our messages**, and we receive theirs. Anything we encode is
readable by the opponent, so a parent-to-child memory code is not private. A
dragon can also hear its own sonar.

### Receiving echoes

    ECHOES <kelp> <ally> <ally_head> <enemy> <enemy_head>

Sits **between the messages and the 49 tile lines**, so it shifts every offset
after it; `tests/blockparse.py` does not know about it and will mis-parse a
protocol-3 block. Present on every turn for a dragon on protocol 3, all zeros
when nothing was sent.

**The five counts sum to exactly the number of rays sent**, so each ray
terminates on exactly one thing and is classified into exactly one category. A
ray does not pass through and tally what it crosses. A kelp edge stops the ray
before an adjacent dragon.

The design consequence: **echoes are one aggregate for the whole turn, with no
direction attached.** Broadcasting in all four directions returns a histogram of
the surroundings and throws the bearing away; sending in *one* direction gives an
unambiguous reading that way. Which is better is empirical, and both are cheap,
so the feature set should allow either.

## Cost

`wasmprobe/meter_bot.sh`, four directed sonars with full 64-bit payloads plus
`PROTOCOL 3` appended to the same buffered write:

    turn            9            10            11
    baseline    68,925,455    68,915,296    68,915,080
    + 4 sonars  69,438,386    69,428,227    69,428,011
    delta          512,931       512,931       512,931

**0.51M points a turn, 0.6% of the cap**, identical every turn, because the lines
join the write the bot already makes rather than adding writes. So "broadcast in
every direction every turn" is affordable. Computing a *useful* payload is the
real cost: the codec's encoder adds ~3.3M, about 3.8M in total.

## Retracted claims

Kept so they are not re-derived:

- ~~"Without kelp we are essentially exact; with kelp we are about 7% out."~~ The
  residual was never mainly about kelp. It was the body-dragging rule, which
  shows up wherever a body is not straight behind the head.
- ~~"The echo path is safe to train on."~~ It was 54.9%–98% by map. It is safe
  now, for a different reason: it is exact.
- ~~"A dragon's own body is transparent to its own sonar."~~ Only the one segment
  behind the head is, and entering it bends the ray.
- ~~"Aggregate message counts agree (191,457 vs 191,455)."~~ They cancelled: over
  on 4,343 turns, under on 4,377. **A total agreeing is not agreement.**
- ~~"A split child starts on the legacy protocol until its own first reply."~~ It
  inherits the parent's.
- ~~"The directed form exists only in protocol 3."~~ It is always accepted.

Hypotheses tested and rejected earlier, all superseded by the body-dragging rule
but recorded so they are not retried: reading the direction letter in the
dragon's own frame; the ray reflecting off kelp; a dragon behind a kelp edge
still being heard; kelp as a fallback rather than a terminator; casting from the
pre-move position; three models of when an inbox is cleared; a stale `head_at`;
duplicate delivery; a different ray length limit; portals.

## Still open

- **Unrelated, pre-existing:** `tests/stress.py` reports 2 block mismatches on
  generated maps (`gen23`, `gen78`, splitter policy) where a split child's
  **facing character** differs (`N` vs `S`). Verified identical at HEAD before
  any of this work, so it is not sonar. See KNOWN_ISSUES.md.
- `tile_after_step` does not rotate the heading when a portal changes edge
  orientation. No official map has such a portal, so it is latent.
- The `empty` echo kind was never observed: on an open torus a ray always comes
  back to its own body, so it always terminates on something.
