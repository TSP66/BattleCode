# Sonar under unswbc 1.0.0 — measured, not guessed

Everything below was read off the reference engine by `bcsim/tests/probe_sonar.py`,
which drives the real WASM engine through `tests/oracle.py` and reports what
comes back. The helper headers give the wire format; they do not give the rules,
and the rules are not what we assumed.

Rerun it with:

    python bcsim/tests/probe_sonar.py ../maps-official/default_small.map

## The thing that gates all of it: `PROTOCOL 3`

The engine speaks the new sonar only to a bot that asks for it, and the shipped
helper asks **on every turn**, immediately before `ENDTURN`:

    PROTOCOL 3
    ENDTURN

This is not a one-time handshake. Without it the engine stays on the old
protocol: no `ECHOES` line, and `SONAR <uint64>` capped at 32 bits.

**`mybot/` does not print `PROTOCOL` at all**, so every version we have
submitted has been running as the legacy protocol. That is why the block format
our `obs.hpp` parses has kept matching and why we have never seen an echo. It
also means adopting protocol 3 changes the block we parse — see below.

Measured: broadcasting in all four directions for 16,024 sonars produced **zero**
echoes and **zero** received messages until `PROTOCOL 3` was added, and then
echoes appeared on 598 of 602 turns.

## Sending

    SONAR <N|E|S|W> <uint64>     directed, full 64 bits
    SONAR <uint32>               legacy, along the current facing only

C++ helper (`templates/cpp/helper.hpp:383`):

```cpp
void send_sonar(Direction direction, std::uint64_t message);
bool send_sonar(std::uint64_t message);   // false if message > UINT32_MAX
```

A dragon may send **one message per cardinal direction in the same turn** — four
separate `SONAR` lines, each with its own payload. Sonar does not consume the
turn's action; it is sent alongside `MOVE` or `SPLIT`.

The full 64 bits survive intact, top bit included: 507 payloads carrying bit 63
and a check byte arrived with **0 corrupted**.

## Receiving messages

    NUM_MSGS <n>
    <uint64>          x n

`get_sonar_messages()` returns them "in the order they were sent". There is **no
sender id and no direction of arrival** — which is exactly why a child needs a
tag byte to recognise its parent.

Measured over one game, messages received per turn: 0 x245, 1 x227, 2 x112,
3 x16, 4 x2. So several can land in one turn, and four is reachable.

By sender: **own 18, ally 255, enemy 234.**

Two consequences:

- **The enemy receives our messages.** Anything we encode is readable by the
  opponent, and we receive theirs. A parent-to-child memory code is not private.
- **A dragon can hear its own sonar**, because a ray can wrap the torus and come
  back to the sender.

## Receiving echoes

    ECHOES <kelp> <ally> <ally_head> <enemy> <enemy_head>

Placed **between the messages and the 49 tile lines** — so it shifts every
offset after it. `tests/blockparse.py` does not know about it and will
mis-parse a protocol-3 block.

```cpp
struct SonarEchoes { int kelp, ally, ally_head, enemy, enemy_head; };
```

The semantics, measured:

- The line is **always present** under protocol 3 (1607 of 1611 turns; absent
  only on the first turns, before anything has been sent).
- **The five counts sum to exactly the number of sonars sent that turn.**
  Sending 0 gives `(0,0,0,0,0)` on all 1607 turns; sending 1 gives a sum of 1 on
  all 1607 turns; sending 4 gives tuples like `(4,0,0,0,0)` and `(3,1,0,0,0)`.
- So **each ray terminates on exactly one thing** and is classified into exactly
  one of the five categories. A ray does not pass through and tally what it
  crosses.
- A kelp edge stops the ray **before** an adjacent dragon: 439 turns had a
  dragon immediately north and still reported kelp.

The single most important consequence for design:

> **Echoes are one aggregate for the whole turn, with no direction attached.**
> Broadcasting in all four directions returns a histogram of what surrounds the
> dragon and throws the bearing away. Sending in *one* direction returns an
> unambiguous reading of what lies that way.

So "send in every direction every turn" buys a 4-way summary; rotating the
direction buys bearings over four turns. Which is better is an empirical
question, and both are cheap, so the feature set should let the policy have
either.

Single-ray readings due north, 1607 turns: kelp 1106, ally body 260, ally head
160, enemy body 43, enemy head 38.

## The sender's own body

**The ray steps out through it, and then it counts again.** Neither of the two
obvious rules is right. Measured on `big_empty`, which has no kelp at all, one
ray per turn, by direction (N is forward, S is straight back down the tail):

| rule | N | E | S | W |
|---|---|---|---|---|
| stop at own body immediately | 0 | 0 | **2974** | 0 |
| never stop at own body | **2992** | **622** | **2992** | **622** |
| leave it, then it counts | **0** | **0** | 18 | **0** |

The kelp-free map is what made this visible: on an open torus a ray with nothing
in its way travels the whole way round and comes back to the dragon that sent
it. Stopping immediately gets the backward ray wrong; never stopping gets the
forward ray wrong, and that ray then finds nothing at all and reports an empty
echo where the engine reports an ally.

The legacy protocol does none of this -- there the ray stops on the sender's own
body at once and the sender receives its own message. `tests/test_vecenv.py`
checks the legacy path against the engine and fails if that is changed.

## Who receives

Delivery is to **any segment**, not only a head: over one game, 501 rays stopped
on a dragon (counted from the echoes) and exactly 501 messages were received.

## Where our simulator still differs

`tests/parity_sonar.py` drives the engine and our simulator in lockstep and
compares every block. The legacy protocol is **byte-identical** on every map
tried. Under protocol 3, driving one ray per turn instead of four localises what
is left -- with four, the echo is a direction-less aggregate and cannot see a ray
going the wrong way:

| map | kelp edges | N | E | S | W | turns |
|---|---|---|---|---|---|---|
| big_empty | 0 | 0 | 0 | 18 | 0 | 3000 |
| default_small | 72 | 111 | 121 | 109 | 126 | 1611 |
| arena | 44 | 10 | 15 | 17 | 5 | 82 |

**Without kelp we are essentially exact; with kelp we are about 7% out.** So
what remains is the ray's interaction with kelp, not its geometry, not its
treatment of the sender, and not the echo categories. The mismatches are almost
entirely one shape: we report `kelp` where the engine reports a dragon
(ours kelp -> engine ally_head x104, ours kelp -> engine ally x103).

Message timing inherits the same residual, and aggregate message counts agree.

Hypotheses tested and **rejected**, recorded so they are not tried again:

- reading the direction letter in the dragon's own frame rather than as a compass
  bearing -- triples the mismatches, and breaks the forward-ray case which is
  exact (that exactness is itself the evidence for absolute, since the two
  readings agree there and nowhere else);
- the ray reflecting off kelp and continuing back -- much worse
  (823/791/1182/1242 on default_small);
- a dragon on the far side of a kelp edge still being heard -- worse
  (293/121/309/466);
- a ray hitting the sender's own head after wrapping while passing through its
  body -- no change;
- casting from the pre-move rather than the post-move position -- worse;
- three different models of when an inbox is cleared.

**So the echo path is safe to train on, and the message path is safe only on maps
without kelp**, which matters because the parent-to-child memory codec rides on
the message path.

## Still unmeasured

- The **point cost** of a `SONAR` line. Four lines a turn is four more writes
  worth of output unless the helper batches them; `SUBMITTING.md` records that
  output is metered at `READ_BYTE_COST` 6/byte and that one flushed write a turn
  is the budgeted case. **Meter this before broadcasting four a turn.**
- Whether the engine's ray has a different length limit than our `w + h`, and
  how it crosses portals. These are the two remaining candidates for the 9%.
- Why 18 messages in one game decoded as coming from the receiver itself, when
  protocol 3 makes a dragon's own body transparent to its own ray.
