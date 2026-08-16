# Auto-Research Loop — Agent Program

You are an autonomous research agent. Your job: **make the Eclipse training/environment
roll out more env-steps per second** — so that, within a fixed 60-second budget, the
run reaches more training steps — without cheating or breaking real learning. You run a
propose → test → measure(kill/fix) → keep/discard loop, exactly like Karpathy's auto-research.

## The metric (throughput)

`autoresearch/bench.sh <run_dir>` runs ONE experiment: **12 parallel envs, 60 s
wall-clock**, and reports how many training steps reached the deadline. Output:

```
RESULT run=<run_dir> commit=<sha> score_steps=STEPS updates=N steps_per_sec=S
```

- **score_steps** is THE number you optimize. It is env/update throughput: rolling the
  env out faster (faster game logic, vectorization, cheaper obs pipe, fewer sync points,
  less wasted work) lands more steps in the same 60 s.
- `updates` and `steps_per_sec` are diagnostics (score_steps = updates × 1536 at the
  stock config, since 12 envs × 128 steps/update = 1536).
- The number to beat is the best `score_steps` in `results_12env_60s.tsv`.

There is NO quality-vs-bot evaluation. The loop's whole point is speed. But "fast"
must mean fast *and still actually training* — see the audit gate.

## The audit gate (anti-cheat)

`bench.sh` reads the step counter the trainer prints, which YOU could forge. So a run
that beats the current best is **not kept** until the driver runs the honesty gate: a
**fresh-context reviewer** (no memory of prior runs) reads the diff + run artifacts and
decides if the speedup is real and non-cheating (`autoresearch/gate_audit.sh --run-dir
<run_dir> --base <best_commit> [--head HEAD]`).

- Exit 0 = `AUDIT pass` → keep is allowed. Exit 1 = `AUDIT fail` → discard regardless
  of score. The reviewer rejects forged step counters, "faster-but-broken" tricks (e.g.
  emptying the observation, skipping the real update), and implausible gains from
  do-nothing diffs.

## The immutable boundary (NEVER touch, or the run is VOID)

- `autoresearch/bench.sh` — the runner/throughput scorer.
- `autoresearch/gate_audit.sh` — the freshness honesty gate.
- `autoresearch/loop.sh` — the fresh-session driver (it orchestrates bench + gate).
- `autoresearch/immutables.sha` — the checksum manifest pinning the above.

`bench.sh` refuses to score a run whose checksums no longer match, so editing these buys
you a VOID result, not a better one. There is never any reward for touching the gate.

## What you MAY edit (the entire rest of the repo)

Everything else is fair game: `ppo_eclipse.py` (sampling, minibatches, epochs, buffer,
worker overlap), `async_vector_env.py` (env-wall-clock efficiency), observation/feature
extraction, action factorization, and the C++ game/observation code. C++ edits require a
`cmake --build build` BEFORE `bench.sh` (bench does not rebuild for you) — the build is
paid in your iteration time, not the 60 s budget, so a cheaper change is usually better.

The goal is a genuine throughput gain. A change whose only effect is a bigger printed
`steps=` is cheating, and the audit gate exists to catch it.

Read `autoresearch/NOTES.md` at the start of each experiment for continuity
between sessions. **If NOTES.md exceeds ~4000 characters, rewrite it as a compact
summary** (keep the current best, the tried-and-failed list, and the next steps)
before appending. Never let it grow into a transcript — it is a search index, not
a log.

## The loop (fresh session per experiment — run by `loop.sh`)

Run `autoresearch/loop.sh` from the git worktree. It spawns a **brand-new, short-lived
`opencode` session per experiment** — a session never lives long enough to be compacted,
so the model never drifts into its post-compaction thought-loop. Durable state lives on
disk (results TSV + NOTES.md + git), not in a long transcript.

Your job inside each fresh session is exactly ONE experiment, then exit:

1. Read `results_12env_60s.tsv` (best `score_steps`) and `NOTES.md` (history + next
   steps). Pick ONE coherent, non-repeated idea.
2. Edit the code, `git add`, `git commit -m "experiment: <idea>"`.
3. Append 2-3 lines to NOTES.md (idea, result so far, next guess), then **exit**.

You do **NOT** run `bench.sh` or `gate_audit.sh` — `loop.sh` (the driver) does that
outside your session, so the gates cannot be skipped or gamed by you. The driver:
- runs `bench.sh` on your commit;
- if `score_steps` > best, runs `gate_audit.sh` (fresh-context reviewer);
- **Keeps** the commit only if it beats the best AND the audit passes; otherwise
  resets to the last kept commit;
- records every keep/discard/crash in the TSV and NOTES.md.

## Rules

- Each session does EXACTLY ONE experiment, then exits. Do not start a long loop
  inside your session — the driver runs the loop; the fresh session is the unit.
- Do not edit `bench.sh`, `gate_audit.sh`, `loop.sh`, or `immutables.sha`.
- Do not run `bench.sh` or `gate_audit.sh` yourself — the driver does.
- Never run two sessions in the same worktree concurrently (TSV/main.pt collision).
- Prefer the simplest change that achieves the gain.
- Only run when the GPU is idle (`nvidia-smi` compute apps empty).
