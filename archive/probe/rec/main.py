"""Plays a fixed legal-ish pattern and records dragon 0's raw stdin, so the
compiled bot can be fed a genuine protocol stream offline."""
import sys, os
OUT = "/tmp/claude-1000/-home-thomaspetty-BattleCode/bf6bf097-99ae-4e03-a3c5-cb400a482746/scratchpad/transcript.txt"
lines = []
first = sys.stdin.readline(); lines.append(first)
ident = int(first.split()[1])
for _ in range(3): lines.append(sys.stdin.readline())
rounds = 0
while True:
    l = sys.stdin.readline()
    if l == "": break
    lines.append(l)
    if l.startswith("ENDGAME"): break
    if l.startswith("ROUND"):
        # read the rest of this round's block up to the blank line
        while True:
            m = sys.stdin.readline(); lines.append(m)
            if m == "\n" or m == "": break
        rounds += 1
        print("MOVE N" if rounds % 4 else "MOVE E"); print("ENDTURN", flush=True)
        if ident == 0 and rounds == 12:
            open(OUT, "w").write("".join(lines) + "ENDGAME\n")
