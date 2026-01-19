# Project Guidelines

## Environment

- Use `uv` for environment management.
- Unless otherwise specified, assume code runs on a single 32GB RTX 5090 GPU server.
- Default assumption: the local machine does not have the full dataset or a comparable high-VRAM GPU.

## Code Quality

- For modified code only, `run ruff check <paths>` must pass with no warnings.
- For modified code only, `run ty check <paths>` must pass with no type errors.
- Do not suppress linter/type checker errors (e.g., `# noqa`, `# type: ignore`) unless it is best practice; if suppression is necessary, add a comment explaining the reason.

## Workflow

- Ask for confirmation before modifying code outside the current discussion scope.
- After each code change, self-review and simplify, then run targeted `ruff`/`ty` checks and ensure they pass.

## Version Control

- Use separate branches or `git worktree` for AI coding changes, based on the task.
- Commit message format: `type: description`
  - `bench:` - benchmark related changes
  - `docs:` - documentation
  - `chore:` - miscellaneous (config, deps, etc.)
  - `fix:` - bug fixes
  - `feat:` - new features (non-benchmark)

## Communication

- Align response language to the user's input language.
