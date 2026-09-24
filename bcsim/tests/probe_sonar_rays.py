"""Per-ray ground truth for sonar, which is the instrument this needed all along.

Aggregate echo counts cannot tell a targeting error from a classification error,
and guessing rules from aggregates has already cost several wrong hypotheses. So
every payload here encodes (round, sender, direction), making each ray globally
unique: the set of messages a dragon receives then names exactly which rays
reached it, and our delivery can be paired with the engine's one ray at a time.

Run it after any change to cast_sonar. The number to watch is "agree", per RAY --
not per turn, and not per echo count.

    python tests/probe_sonar_rays.py

Measured 2026-09-24, four rays a turn, protocol 3:

    big_empty      97.20% of 191,463 rays    (no kelp, no portals)
    default_small  87.45% of     518 rays
    arena          69.23% of      91 rays
"""
import pathlib, sys, collections
sys.path.insert(0,"bcsim/tests")
import parity_sonar as P

TOP = 1 << 63
def tag(rnd, sender, d): return TOP | (rnd & 0xFFFF) << 24 | (sender & 0xFFFF) << 8 | (d & 0x3)
def untag(v):
    if not (v & TOP): return None
    return ((v >> 24) & 0xFFFF, (v >> 8) & 0xFFFF, v & 0x3)

def run(map_text, protocol=3):
    sim = P.TextSim(map_text)
    ours, eng = {}, {}            # ray -> receiver
    team = {}                     # dragon -> team letter, from the bodies list
    def policy(did, text):
        b = P.parse(text)
        for v in b["msgs"]:
            u = untag(v)
            if u: eng.setdefault(u, []).append(did)
        di = sim.next()
        if di >= 0 and sim.dragon_id(di) == did:
            try: o = P.parse(sim.block(di))
            except Exception: o = None
            if o is not None:
                for v in o["msgs"]:
                    u = untag(v)
                    if u: ours.setdefault(u, []).append(did)
        r = b["round"]
        reply = [f"SONAR {d} {tag(r,did,k)}" for k,d in enumerate(P.DIRS)]
        ok = P.safe_dirs(b)
        if b["length"] >= 6 and (r % 7) == 0: reply.append(f"SPLIT {b['length']//2}")
        else: reply.append(f"MOVE {ok[0] if ok else b['dir']}")
        reply += [f"PROTOCOL {protocol}", "ENDTURN"]
        out = "\n".join(reply) + "\n"
        if di >= 0: sim.reply(di, out)
        return out
    try: P.OracleGame(map_text, policy).run()
    except Exception as e: print("  err", type(e).__name__, e)
    sim.close(); return ours, eng

def rel(recv, sender): return "SELF" if recv == sender else "other"

def by_turn(d):
    """(round, sender) -> sorted multiset of receivers, direction label ignored."""
    out = {}
    for (rnd, sender, dirn), who in d.items():
        out.setdefault((rnd, sender), []).extend(who)
    return {k: sorted(v) for k, v in out.items()}

for name in ("default_small", "big_empty", "arena"):
    mp = pathlib.Path("maps-official")/f"{name}.map"
    ours, eng = run(mp.read_text())
    rays = set(ours) | set(eng)
    cls = collections.Counter()
    for ray in rays:
        rnd, sender, d = ray
        o = ours.get(ray, []); e = eng.get(ray, [])
        ov = o[0] if o else None
        ev = e[0] if e else None
        if ov == ev: cls["agree"] += 1
        else:
            a = "nobody" if ov is None else rel(ov, sender)
            c = "nobody" if ev is None else rel(ev, sender)
            cls[f"ours={a:6s} engine={c}"] += 1
    tot = sum(cls.values())
    print(f"\n{name}: {tot} distinct rays that delivered to someone")
    for k, c in cls.most_common(10):
        print(f"    {k:34s} {c:6d}  ({100*c/max(tot,1):5.2f}%)")
    # Same question ignoring WHICH direction each ray was labelled with. If this
    # agrees where the per-direction view does not, the geometry is right and only
    # the direction->payload association differs.
    import collections as _c
    do = _c.Counter(len(v) for v in ours.values())
    de = _c.Counter(len(v) for v in eng.values())
    print(f"    times each unique ray was RECEIVED -- ours {dict(sorted(do.items()))}"
          f"  engine {dict(sorted(de.items()))}")
    ot, et = by_turn(ours), by_turn(eng)
    ks = set(ot) | set(et)
    agree = sum(1 for k in ks if ot.get(k, []) == et.get(k, []))
    print(f"    ignoring the direction label: {agree}/{len(ks)} sender-turns agree "
          f"({100*agree/max(len(ks),1):.2f}%)")
