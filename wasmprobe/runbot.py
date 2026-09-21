"""Runs the compiled C++ bot under the judge's metering, turn by turn.

`unswbc run --sandbox` only prices Python bots, and a submission reports
nothing back but the result. This closes the gap: the bot is compiled to
wasm32-wasi with the judge's flags, instrumented with the toolkit's copy of
the judge's metering pass, and fed a recorded protocol transcript.

Two WASI calls are shadowed, both as the judge's sandbox defines them:
  * fd_write charges 2.5M points plus 4000 a byte, and an ENDTURN on stdout
    marks the end of a turn, which is where the per-turn cost is read off;
  * clock_time_get returns points spent, so the bot's own stage timings
    come out in points too.

    python runbot.py bot.wasm transcript.txt ../mybot
"""

from __future__ import annotations

import os
import struct
import pathlib
import sys

sys.path[:0] = [str(p) for p in pathlib.Path.home().glob(
    ".local/share/uv/tools/unswbc/lib/python3*/site-packages")]

from unswbc import metering                                        # noqa: E402
from wasmtime import (Engine, FuncType, Linker, Module, Store,     # noqa: E402
                      ValType, WasiConfig)

TURN_LIMIT = 100_000_000
WRITE_SYSCALL = 2_500_000
WRITE_BYTE = 4_000
REMAINING = "wasmer_metering_remaining_points"


def main() -> None:
    wasm, transcript, botdir = sys.argv[1], sys.argv[2], sys.argv[3]
    initial = metering.INITIAL_POINTS
    blob = metering.instrument(open(wasm, "rb").read(), initial)

    engine = Engine()
    module = Module(engine, blob)
    store = Store(engine)
    wasi = WasiConfig()
    wasi.stdin_file = transcript
    # the judge mounts nothing, so by default neither do we; NO_FS=0 restores
    # the old behaviour of exposing the bot's directory as "."
    if os.environ.get("NO_FS", "1") == "0":
        wasi.preopen_dir(botdir, ".")
    wasi.argv = ["bot"]
    store.set_wasi(wasi)

    linker = Linker(engine)
    linker.allow_shadowing = True
    linker.define_wasi()

    state = {"out": bytearray(), "turn_start": 0, "turns": [], "meter": None}

    def spent(caller) -> int:
        return initial - caller[REMAINING].value(caller)

    def charge(caller, points: int) -> None:
        g = caller[REMAINING]
        g.set_value(caller, max(0, g.value(caller) - points))

    def fd_write(caller, fd, iovs, iovs_len, nwritten):
        mem = caller["memory"]
        raw = mem.read(caller, iovs, iovs + 8 * iovs_len)
        data = bytearray()
        for i in range(iovs_len):
            ptr, ln = struct.unpack_from("<II", raw, 8 * i)
            data += mem.read(caller, ptr, ptr + ln)
        mem.write(caller, struct.pack("<I", len(data)), nwritten)
        if fd == 1 and data:
            charge(caller, WRITE_SYSCALL + WRITE_BYTE * len(data))
            state["out"] += data
            text = bytes(data)
            while b"ENDTURN" in text:
                now = spent(caller)
                state["turns"].append(now - state["turn_start"])
                state["turn_start"] = now
                text = text.split(b"ENDTURN", 1)[1]
        elif fd == 2:
            sys.stderr.write(data.decode(errors="replace"))
        return 0

    def clock_time_get(caller, clock_id, precision, out):
        caller["memory"].write(caller, struct.pack("<Q", spent(caller)), out)
        return 0

    i32, i64 = ValType.i32(), ValType.i64()
    linker.define_func("wasi_snapshot_preview1", "fd_write",
                       FuncType([i32, i32, i32, i32], [i32]), fd_write, access_caller=True)
    linker.define_func("wasi_snapshot_preview1", "clock_time_get",
                       FuncType([i32, i64, i32], [i32]), clock_time_get, access_caller=True)

    inst = linker.instantiate(store, module)
    err = None
    try:
        inst.exports(store)["_start"](store)
    except Exception as e:          # proc_exit and traps both land here
        err = e

    out = state["out"].decode(errors="replace")
    if os.environ.get("OUT_FILE"):
        open(os.environ["OUT_FILE"], "w").write(out)
    logs = [l for l in out.splitlines() if l.startswith("LOG")]
    print(f"{'turn':>5s} {'points':>14s}  {'% of limit':>10s}")
    for i, pts in enumerate(state["turns"]):
        flag = "  OVER" if pts > TURN_LIMIT else ""
        print(f"{i:5d} {pts:14,}  {pts / TURN_LIMIT:10.1%}{flag}")
    print("\nbot's own stage log (points since turn start):")
    for l in logs[:6]:
        print("  " + l)
    if err is not None and "exit status 0" not in str(err):
        print(f"\nstopped: {type(err).__name__}: {str(err).splitlines()[0]}")


if __name__ == "__main__":
    main()
