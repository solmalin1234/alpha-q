# Experiments

## Conventions

Each experiment lives in its own subdirectory named by date and short description:

```
experiments/
  2024-01-15_pong-vanilla-dqn/
    notes.md          # Hypothesis, config changes, observations
    config.yaml       # Exact config used (or overrides)
    results.md        # Final metrics, plots, conclusions
```

## Workflow

1. Create a directory: `experiments/YYYY-MM-DD_short-description/`
2. Write `notes.md` with your hypothesis and what you're testing
3. Run training with the appropriate config overrides
4. Record results in `results.md` with MLFlow run IDs
5. Compare runs in the MLFlow UI: `make mlflow`

## MLFlow

All training runs are logged to MLFlow automatically. Use `make mlflow` to launch the UI.

Tracked metrics:
- `episode/reward` — raw episode return
- `episode/length` — episode length in steps
- `train/loss` — TD loss
- `train/q_mean` — mean predicted Q-value
- `train/epsilon` — current exploration rate
- `eval/reward_mean` — mean evaluation reward (over N episodes)
- `eval/reward_std` — std of evaluation rewards
