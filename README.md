# Alpha-Q

Deep Q-Network variants for Atari — from vanilla DQN to Rainbow and beyond.

A research project implementing the progression of DQN improvements from the deep RL literature, with each variant built incrementally and measured via controlled experiments.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd alpha-q

# Install all dependencies (creates .venv automatically)
uv sync --extra dev

# Or use the Makefile shortcut
make install
```

For Atari ROMs, `ale-py` handles ROM installation automatically.

## Usage

### Train an agent

```bash
uv run python scripts/train.py                                        # defaults (Pong, DQN)
uv run python scripts/train.py --env configs/envs/breakout.yaml       # different env
uv run python scripts/train.py --set training.total_steps=500000      # override any config

# Or via Make
make train ARGS="--env configs/envs/pong.yaml"
```

### Watch a trained agent play

```bash
uv run python scripts/play.py --checkpoint checkpoints/agent_final.pt
```

### Record gameplay videos

```bash
uv run python scripts/record.py --checkpoint checkpoints/agent_final.pt --episodes 3
```

### MLFlow experiment tracking

```bash
make mlflow   # opens UI at http://localhost:5000
```

## Configuration

Configs are layered YAML files merged in order: `default -> agent -> env -> CLI overrides`.

```
configs/
├── default.yaml           # all defaults
├── agents/dqn.yaml        # agent-specific overrides
└── envs/
    ├── pong.yaml          # Pong environment
    └── breakout.yaml      # Breakout environment
```

Override any value from the command line with `--set key.subkey=value`.

## Project Structure

```
alpha-q/
├── pyproject.toml          # Project metadata + dependencies (uv/hatch)
├── Makefile                # Convenience targets
├── configs/                # YAML configuration files
├── experiments/            # Experiment logs and notes
├── scripts/                # CLI entry points (train, play, record)
├── src/alpha_q/
│   ├── agents/             # Agent registry + implementations
│   ├── networks/           # Neural network architectures (Nature CNN, etc.)
│   ├── memory/             # Replay buffers (uniform, prioritized)
│   ├── envs/               # Environment factories + wrappers
│   ├── training/           # Training loop + evaluation
│   └── utils/              # Config loading, MLFlow logging, seeding
└── tests/                  # Unit and smoke tests
```

## Key Design Decisions

- **uv** for fast, reproducible dependency management with lockfile
- **uint8 observations** in replay buffer, normalized to float in the network forward pass (4x memory savings)
- **Layered YAML configs** with deep merge — no heavy framework deps like Hydra
- **Agent registry** pattern — `create_agent("dqn", ...)` — easy to add new variants
- **MLFlow** for experiment tracking: params, metrics, model checkpoints, gameplay videos

## Research Roadmap

Each phase is a separate branch and PR with documented experiment results.

| Phase | Variant | Paper | Key Change |
|-------|---------|-------|------------|
| 1 | Vanilla DQN | Mnih et al., 2015 | Full DQN agent, train on Pong |
| 2 | Double DQN | van Hasselt et al., 2016 | Decouple action selection/evaluation |
| 3 | Prioritized Replay | Schaul et al., 2016 | Sum-tree PER + importance sampling |
| 4 | Dueling DQN | Wang et al., 2016 | Value + advantage stream split |
| 5 | Multi-step Returns | Sutton & Barto | N-step returns (n=3) |
| 6 | Noisy Nets | Fortunato et al., 2018 | NoisyLinear, remove epsilon-greedy |
| 7 | C51 (Distributional) | Bellemare et al., 2017 | 51-atom value distribution |
| 8 | Rainbow | Hessel et al., 2018 | Compose all 6 improvements |
| 9 | QR-DQN | Dabney et al., 2018 | Quantile regression replaces fixed atoms |
| 10 | Munchausen DQN | Vieillard et al., 2020 | Scaled log-policy bonus in Bellman target |
| 11 | Benchmarks | — | Learning curves across agents and games |

### After Phase 11

- [ ] **Ablation studies** — Systematically remove each Rainbow component to measure individual contribution (reproduce Table 2 from Hessel et al.)
- [ ] **Munchausen + Rainbow hybrid** — Combine M-DQN's log-policy bonus with Rainbow; measure if entropy regularisation can replace NoisyNets
- [ ] **QR-Rainbow** — Swap C51's fixed atoms for QR-DQN quantiles inside Rainbow; compare distributional approaches head-to-head
- [ ] **Per-game hyperparameter tuning** — Sweep `v_min`/`v_max`, `n_step`, `alpha`/`beta` per environment; document which matter most
- [ ] **Scaling to full Atari-57** — Run best agents on the standard 57-game benchmark suite, compare to published scores
- [ ] **Write-up** — Compile results into a report or blog post with learning curves, ablation tables, and key takeaways

## Development

```bash
make test     # run tests
make lint     # check style (ruff)
make format   # auto-format
```
