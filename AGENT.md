# Project Guidelines

## Environment

- Use `uv` for environment management

## Code Quality

- Code must pass `uv run ruff check` with no warnings
- Code must pass `uv run ty check` with no type errors
- Do not suppress linter/type checker errors (e.g., `# noqa`, `# type: ignore`) unless it is the best practice; if suppression is necessary, add a comment explaining the reason

## Workflow

- Ask for confirmation before modifying code outside the current discussion scope
