# Contributing

Thanks for your interest. This repository is primarily an **archival reproducibility
package** for a published paper, so the bar for changes differs from an ordinary project.

## What we welcome

- Bug reports where a script fails to run on a documented environment.
- Portability fixes (newer Python, different BLAS, Windows path handling).
- Documentation clarifications.
- Independent replication reports — please open an issue with your environment details
  and the output of `python scripts/verify_raw_results.py`.

## What we will not merge

- Changes that alter the numerical results archived in `results/`. Those values back a
  published paper and are frozen. If you find a genuine methodological error, open an
  issue rather than a pull request so it can be handled through the journal's correction
  process.
- New model variants or extensions. Please fork instead — we will happily link to your work.

## Before opening a pull request

1. Confirm `python scripts/verify_raw_results.py` still passes.
2. Note your Python, PyTorch, NumPy and OS versions in the PR description.
3. Keep diffs minimal and focused on one issue.
