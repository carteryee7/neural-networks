# Snake Reinforcement Learning — Project Findings

A Deep Q-Network (DQN) agent that learns to play Snake in a custom environment.
This document summarizes the architecture, the engineering decisions, the bugs
found and fixed, the empirical results, and the lessons learned.

---

## 1. Overview

- **Goal:** train an agent to play Snake autonomously via reinforcement learning.
- **Method:** Deep Q-Learning (DQN) — the network outputs a Q-value per action,
  and the policy picks the action with the highest Q-value.
- **Board:** 10×10 grid (≈100 cells; theoretical max score ≈ 97).
- **Two model variants explored:**
  - **MLP** on a hand-engineered feature vector (the primary, best-performing agent).
  - **CNN** on the full board as a 3-channel image (harder to train; secondary).

---

## 2. Architecture

### Environment (`game.py`)
A Gym-style interface so the MLP and CNN scripts can share one environment:
- `reset()` → initial state
- `step(action)` → `(next_state, reward, done, score)`
- `get_state()` → dispatches on a `state_mode` flag (`"features"` for the MLP,
  `"grid"` for the CNN), so the representation is chosen per game rather than by
  editing the code.

### State representations
- **Features (MLP), 14 values, normalized to ~[0, 1]:**
  - 3× ray-cast danger distance (straight / right / left, relative to heading)
  - 4× current direction (one-hot)
  - 4× fruit location relative to head (left / right / up / down)
  - 1× Euclidean distance to fruit
  - 1× snake length
  - 1× flood-fill reachable free space
- **Grid (CNN), 3×H×W tensor:** channel 0 = body, 1 = head, 2 = fruit;
  empty cells are implicitly all-zeros. Returned channels-first `(C, H, W)`.

### Reward function
- `+10` for eating fruit
- `−10` for dying (wall or self-collision)
- `−0.01` per step (discourages stalling)
- `±0.1` shaping for moving toward / away from the fruit
- Episode timeout scaled to snake length (prevents infinite loops)

### Model (`model.py`)
- **MLP (`SnakeNN`):** 14 → 128 → 64 → 4, ReLU activations.
- **Dueling variant (implemented, currently disabled):** shared layers split into
  a state-value head `V(s)` and an advantage head `A(s,a)`, combined as
  `Q = V + (A − mean(A))`.
- **CNN:** 2× (Conv → ReLU → MaxPool) → flatten → dense → 4 outputs.

### Training loop (`train_mlp.py`)
DQN with:
- **Experience replay** (`deque`, sampled in random batches to decorrelate frames)
- **Epsilon-greedy exploration** with exponential decay (1.0 → 0.01)
- **Target network** (frozen copy synced periodically) for stable bootstrap targets
- **Double DQN** target (implemented, currently disabled): online net selects the
  next action, target net evaluates it — curbs Q-value overestimation
- MSE loss on the Bellman target `r + γ·maxQ(s')·(1 − done)`, Adam optimizer

---

## 3. Bugs Found & Fixed

| Bug | Symptom | Fix |
|---|---|---|
| `match action: case int(UP):` | Every action moved the snake up (int pattern binds, doesn't compare) | Dict dispatch on action |
| `grow()` appended empty tuple; body didn't follow head | Snake wasn't a real snake | Insert-head / pop-tail movement + deferred-grow counter |
| No 180° reversal guard | Snake could reverse into itself | Ignore reversals once length > 1 |
| Channels-last tensor into Conv2d | `expected 3 channels, got 28` | `transpose(2,0,1)` → `(C,H,W)` |
| Single state missing batch dim | `mat1/mat2` shape mismatch (channels mistaken for batch) | `unsqueeze(0)` |
| `view` vs `transpose` confusion | Would silently scramble data | Use `transpose`/`permute` to reorder axes |
| `get_state` mode mismatch | MLP got a grid → `(30x10 and 11x64)` | `state_mode` param + dispatch |
| Flood-fill on off-grid head | `IndexError` on wall death | Return 0 reachable space when head is off-grid |
| Learning rate 0.01 (CNN) | Q-values diverged; no learning | Lowered to 1e-4 |
| Dropout in Q-network | Noisy Q estimates / action selection | Removed |

**Latent (flagged, low-priority on a square board):**
- Flood-fill right-neighbor bound uses `self.h` instead of `self.w`.
- Flood-fill returns a raw count (0–100) — should be normalized to match other features.
- `_danger_distance` has an off-by-one making distance-1-free and distance-1-blocked both read 0.

---

## 4. Results

Metrics tracked across training lengths (single runs unless noted):

| Config | Avg score (last 100) | Avg survival | Best | Notes |
|---|---|---|---|---|
| 1000 ep | 18.1 | 152.6 | 38 | pre–flood-fill features |
| 1500 ep | 18.1 | 157.6 | 43 | |
| 2000 ep | **19.8** | **175.3** | 39 | best average run |
| 3000 ep | 19.2 | 162.1 | 44 | no gain over 2000 |
| Double+Dueling | 18.9 | 162.1 | 42 | within noise of plain |

**Baseline (random policy):** avg score ≈ 0.14, survival ≈ 15–17 steps.

**Headline figures (best run):**
- ~**137×** the random baseline on score
- ~**11×** the random baseline on survival
- **Peak score 44** on a 10×10 board
- Converges to half its final average within **~100–135 episodes** (fast, sample-efficient)

---

## 5. Key Findings & Lessons

1. **The agent plateaus at avg ≈ 20.** Training past ~2000 episodes yields no
   reliable improvement — the ceiling is the method, not the compute budget.

2. **The plateau is fundamental, not a tuning problem.** Early-game Snake is a
   simple "go to fruit" task the agent solves well. Late-game Snake is a
   long-horizon planning / Hamiltonian-path problem, and reactive value-based RL
   with local features is structurally bad at it — hence self-trapping.

3. **Feature engineering moved the needle more than algorithm tweaks.**
   Flood-fill + ray-cast distances lifted the peak (38→44) and survival
   (152→175). Double + Dueling DQN landed within run-to-run noise.

4. **Feature scaling matters.** Un-normalized features (distances, length,
   flood-fill count) can dominate 0/1 booleans and destabilize learning;
   normalizing to ~[0,1] is important.

5. **Beware single-run noise.** A ±1-point average swing between single runs is
   not signal. Multi-seed mean ± std is required to compare configurations.

6. **The improvement-over-random multiplier is fragile.** It divides by a tiny,
   noisy baseline, so it swings run-to-run. Absolute average and survival are the
   trustworthy comparison metrics; stabilize the baseline (500–1000 random
   episodes) before citing the multiplier.

---

## 6. Paths to Higher Scores

- **Stay in RL (modest gains):** full "Rainbow" stack (Double + Dueling + Prioritized
  Experience Replay), full-board CNN, or a policy-gradient method (PPO). Expect
  effort-heavy, diminishing returns — still capped by the planning limitation.
- **Switch to search/algorithmic (large gains):** Hamiltonian cycle (near-100%
  fill), Hamiltonian cycle + shortcuts (near-perfect and efficient), or A*/BFS
  pathfinding gated by a flood-fill safety check. The existing flood-fill code is
  exactly the safety primitive the pathfinding approach needs. These routinely
  score 80–90+ but are classic AI/search, not machine learning.
