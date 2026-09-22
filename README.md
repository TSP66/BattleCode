# BattleCode

PPO self-play for UNSW Battlecode 2026 ("dragons"), trained against a fast C++
simulator and shipped as a C++ bot with the weights compiled in.

| Path | What |
|---|---|
| `bcsim/cpp/` | The simulator (`bc_core.hpp`), observations, vectorised env, scripted yardstick bots |
| `bcsim/bcsim/` | Python ctypes wrapper (`BattlecodeVecEnv`); `make -C bcsim` builds the `.so` files here |
| `bcsim/train/` | `train.py` (PPO), `yardstick.py` (eval worker), `dash.py` (dashboard), `augment.py` (map variants), `export_cpp.py` |
| `bcsim/tests/` | Parity tests against the real engine, reward tests |
| `maps/` | Training/eval maps (`maps-official/` = the official set as downloaded) |
| `runs/anchors/`, `runs/submitted/` | Frozen opponents the yardstick always plays (committed); everything else in `runs/` is ignored |
| `mybot/` | The C++ submission. `weights_data.hpp` is generated, not committed |
| `wasmprobe/` | Judge-cost metering, parity checks, `check_bot.sh`, `submit.sh` |
| `archive/` | Old probes and Python bots, kept for reference |
| `runs/replays/` | Scraped games of the top teams (datasets are rebuilt, see `DISTILL.md`) |

**Distillation on another machine: see `DISTILL.md`.** Current RL state: `HANDOFF.md`.

## Setup (new machine)

Needs `uv`, `g++` and an NVIDIA driver.

```bash
git clone <repo> BattleCode && cd BattleCode
scripts/setup.sh        # .venv-train (torch cu128 + numpy), builds the simulator, smoke test
```

For an older GPU/driver, pick another wheel index:
`TORCH_INDEX=https://download.pytorch.org/whl/cu126 scripts/setup.sh`.

## Running a training run

```bash
scripts/launch.sh v5_lr1e-4 --width 64 --blocks 4 --envs 1024 --steps 256 \
    --minibatch 8192 --compile --iters 200000 --reward v4 --lr 1e-4
```

That starts `train.train` and `train.yardstick` in the background, writing to
`runs/<name>/`. Set `DASH_PORT=8770` to start the dashboard too; the URL, including its
access token, goes to `runs/dash.log`. See `python -m train.train -h` (from `bcsim/`)
for every flag. The ones worth sweeping are `--lr`, `--ent`/`--ent-end`/`--ent-half-life`,
`--gamma`, `--lam`, `--clip`, `--epochs`, `--minibatch`, `--reward`, `--size-alpha`
and `--aug-per-map`.

- **Width 64 × 4 blocks is the largest net that fits the judge's CPU budget.**
  Anything bigger can't be submitted as it stands (see `SUBMITTING.md`).
- `--threads` (default 16) is the simulator's thread count, so set it to the
  machine's core count.
- Resume: rerun the same command with `--resume runs/<name>/latest.pt`.
- Bring results back by copying `runs/<name>/` (`log.jsonl`, `eval.jsonl`,
  `snapshots/`) to this machine.

The reward versions (`v1`–`v4`) are defined at the top of `bcsim/train/train.py`.

## Rebuilding the simulator

`make -C bcsim`. It builds to a temp file and moves it into place, so it's safe while
a run is live. Never write over `libbcvec.so` in place: a running process has it mmapped.

## Submitting

C++ only. Use `wasmprobe/submit.sh <ckpt> "description"`. See `SUBMITTING.md` and
`KNOWN_ISSUES.md`.
