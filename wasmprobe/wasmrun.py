"""Runs a wasm32-wasi bot as a native one would run: stdin and stdout are this process's, and
stderr goes to $BC_DUMP_FILE (a -DBC_DUMP build writes its dump there, having no files of its
own). For parity_ffl.py, which takes an executable: `wasmrun.sh bot.wasm`.

    BC_DUMP_FILE=dump.txt python wasmrun.py bot.wasm < transcript
"""
import os
import pathlib
import sys

sys.path[:0] = [str(p) for p in pathlib.Path.home().glob(".local/share/uv/tools/unswbc/lib/python3*/site-packages")]
from wasmtime import Engine, Linker, Module, Store, WasiConfig  # noqa: E402

eng = Engine()
mod = Module.from_file(eng, sys.argv[1])
lk = Linker(eng)
lk.define_wasi()
st = Store(eng)
w = WasiConfig()
w.inherit_stdin()
w.inherit_stdout()
if os.environ.get("BC_DUMP_FILE"):
    w.stderr_file = os.environ["BC_DUMP_FILE"]
else:
    w.inherit_stderr()
st.set_wasi(w)
inst = lk.instantiate(st, mod)
try:
    inst.exports(st)["_start"](st)
except Exception as e:  # proc_exit(0) surfaces as an exit trap
    if "exit status 0" not in str(e):
        raise
