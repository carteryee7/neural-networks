# neural-networks

A collection of small machine learning projects built from scratch in PyTorch,
progressing from basic classifiers to a reinforcement learning agent.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Projects

### 🐍 Snake RL (`snakeNN/`)
A Deep Q-Network (DQN) agent that learns to play Snake in a custom Gym-style
environment. Implements experience replay, epsilon-greedy exploration, a target
network, and an engineered 14-feature state (including ray-cast danger distances
and a flood-fill free-space signal), plus optional Double + Dueling DQN. The
best agent reaches ~137× a random baseline (peak 44 on a 10×10 board). A full
write-up of the architecture, bug fixes, results, and lessons is in
[`snakeNN/FINDINGS.md`](snakeNN/FINDINGS.md).

### 🔢 Handwritten Digits — CNN (`convolutional/`)
A convolutional network classifying handwritten digits, with a Pygame drawing
tool to test predictions on your own input. Includes tuning work: fixing a
too-high learning rate, correcting train/eval mode handling, and early stopping.

### 🔢 Handwritten Digits — MLP (`digits/`)
A fully-connected baseline (784 → 128 → 64 → 10) for the same digit task —
useful as an ANN comparison point against the CNN above.

### 🌸 Iris Classifier (`iris/`)
A small MLP (4 features → 3 classes) classifying iris flowers — the classic
intro classification problem.

### 🎲 Probability Simulations (`random_sim.py`, `random_test.py`)
Monte Carlo dice/random-distribution simulations exploring probability empirically.

### 🧠 Neural Network from Scratch (`scratch/`)
A neural network built entirely in NumPy — no autograd, no PyTorch. Implements
custom `Linear` layers with He initialization, forward propagation, and manual
backpropagation (ReLU, softmax, cross-entropy) trained on the handwritten-digit
task (784 → 128 → 64 → 10). Demystifies what frameworks do under the hood by
deriving and coding the gradients by hand.
