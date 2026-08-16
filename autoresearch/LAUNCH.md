Start the auto-research loop from autoresearch/program.md.

Set up exactly one git worktree for the loops (check if it already exists):

git worktree add ../ar_wt -b ar/speed master
cd ../ar_wt

Run a fresh baseline once (establishes the score_steps to beat):

./autoresearch/bench.sh /tmp/ar_wt_base 12 60

Then run the fresh-session driver loop (from the worktree root):

./autoresearch/loop.sh --experiments 100

`loop.sh` spawns a brand-new, short-lived `opencode` session per experiment so no
session ever lives long enough to be compacted (the model loops after compaction —
fresh sessions are the fix). Each session picks one idea, edits, commits, and exits.
The driver then runs bench.sh (measure) and, on a would-be keep, gate_audit.sh
(fresh-context honesty review). Keep = beat best AND audit pass; else discard.
Results land in results_12env_60s.tsv; search state lives in autoresearch/NOTES.md.

Leave autoresearch/{bench.sh,gate_audit.sh,loop.sh,immutables.sha} untouched.
Use only this worktree. Stop the driver (Ctrl-C) when you want to.
