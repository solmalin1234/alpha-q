# Agent Guidelines

## Git Branching Strategy

**Never commit directly to `develop` or `main`.**

1. **Feature/fix branches**: Always create a `feat/` or `fix/` branch off `develop` for your work.
2. **Merge to develop**: Once the feature branch is ready, merge it into `develop` (typically via PR).
3. **Merge to main**: Only merge `develop` into `main` when explicitly requested by the user.

```
feat/my-feature  →  develop  →  main
fix/my-bugfix    →  develop  →  main
```

## Commit Message Format

Use conventional commit prefixes:

```
feat: add replay buffer with prioritized sampling
fix: prevent recent tiles selection ring from clipping
refactor: simplify DQN target network update logic
docs: update readme with training instructions
test: add unit tests for epsilon-greedy policy
chore: update dependencies
```

- Keep the subject line under 72 characters
- Use imperative mood ("add", "fix", "update", not "added", "fixes", "updated")
- Optionally add a body after a blank line for more detail
