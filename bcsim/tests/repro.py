"""Pinpoints the first turn where our sim and the engine disagree."""
import random, sys
import diffsim, policies
from oracle import OracleGame


def find(map_text, policy, label=""):
    sim = diffsim.Sim(map_text)
    history = []

    def bridge(dragon_id, block):
        di = sim.next_turn()
        ours_id = sim.dragon_id(di) if di >= 0 else None
        if di < 0 or ours_id != dragon_id:
            raise SystemExit(f"turn order diverged: engine {dragon_id}, ours {ours_id}\n"
                             + dump(history))
        ours = sim.round_block(di)
        if ours != block:
            raise SystemExit(f"block diverged at turn {len(history)} dragon {dragon_id}\n"
                             + diffsim._diff(block, ours, label) + "\n" + dump(history))
        text = policy(dragon_id, block)
        sim.reply(di, text)
        history.append((dragon_id, block, text))
        # deaths must match immediately after each action
        if sim.deaths() != [d for d in game.deaths if d not in seen]:
            pass
        return text

    seen = []
    game = OracleGame(map_text, bridge)
    game.run()
    print("no divergence")


def dump(history, n=3):
    out = []
    for dragon_id, block, reply in history[-n:]:
        out.append(f"--- turn: dragon {dragon_id} replied {reply!r}\n{block}")
    return "\n".join(out)


if __name__ == "__main__":
    mp = sys.argv[1]
    seed = int(sys.argv[2])
    pol = getattr(policies, sys.argv[3])
    find(open(mp).read(), pol(random.Random(seed)), label=mp)
