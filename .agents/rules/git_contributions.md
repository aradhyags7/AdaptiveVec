# Git Contribution & Continuous Push Policy

1. **Continuous Incremental Push (No Mega-Batches):**
   - Stage, commit, and push immediately after every granular task or logical improvement.
   - Never hoard 10–25 commits to push in a single batch.
   - Every single commit must be followed immediately by `git push origin main` so GitHub's webhook registers real-time individual contribution events without hitting batch aggregation queues.

2. **Maximum Contribution Cadence:**
   - Decompose work into clear, atomic commits (e.g., separate commits for data extraction, C++ algorithms, React UI views, CSS design tokens, benchmarks, test suites, and documentation).
   - Keep commit messages formatted with Conventional Commits (`feat:`, `fix:`, `assets:`, `docs:`, `perf:`, `test:`).
   - Always verify authorship is locked to `Aradhya Shinde <aradhyashinde2330@gmail.com>`.

3. **Direct to Default Branch:**
   - Always commit and push directly to `main` so that all contributions are counted immediately by GitHub.
