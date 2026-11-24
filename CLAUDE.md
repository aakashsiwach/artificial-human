# Claude Code instructions

@AGENTS.md

In addition:
- Complete the requested implementation rather than stopping after a plan, unless the user explicitly asks only for planning or research.
- Before editing, identify relevant existing modules and tests; do not create duplicate subsystems.
- Use current repository code as the baseline and preserve uncommitted changes.
- For neuroscience research, cite and record the actual papers, datasets, publication dates, and limitations used to motivate a design.
- Keep an implementation ledger: distinguish what is merged, what is in the current branch, and what is only proposed.
- Do not infer that a PR is merged from a local successful test run. Verify GitHub state and CI.
