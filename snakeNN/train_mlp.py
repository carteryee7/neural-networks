import pygame
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from game import SnakeGame
from model import SnakeNN
from collections import deque
import random
import copy
import matplotlib.pyplot as plt

rows = 10
cols = 10

CELL_SIZE = 70

pygame.init()
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
clock = pygame.time.Clock()
game_font = pygame.font.Font(None, 20)

torch.manual_seed(67)
model = SnakeNN()

target_model = copy.deepcopy(model)
target_model.eval()
target_update_freq = 1000
step_count = 0

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

game = SnakeGame(rows, cols, CELL_SIZE) # featues default
episodes = 2000
gamma = 0.9
epsilon = 1.0
epsilon_decay = .95
epsilon_min = .01

memory = deque(maxlen=100_000)
batch_size = 64
scores = []
survival = []   # steps survived per episode


def evaluate_random(n_episodes=500):
    """Average score/survival of a pure-random policy — the learning baseline."""
    eval_game = SnakeGame(rows, cols, CELL_SIZE)
    total_score = total_steps = 0
    for _ in range(n_episodes):
        eval_game.reset()
        steps = 0
        while not eval_game.done:
            _, _, _, s = eval_game.step(random.randint(0, 3))
            steps += 1
        total_score += s
        total_steps += steps
    return total_score / n_episodes, total_steps / n_episodes


baseline_score, baseline_steps = evaluate_random(100)
print(f"Random baseline -> avg score: {baseline_score:.2f}, avg survival: {baseline_steps:.1f} steps")

for i in range(episodes):

    state = torch.tensor(game.reset())
    steps = 0

    while not game.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.done = True
        
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            with torch.no_grad():
                pred = model(state)
                action = pred.argmax().item()
        
        next_state, reward, done, score = game.step(action)
        steps += 1

        memory.append((state, action, reward, next_state, done))

        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)
            states      = torch.tensor(np.array(states))
            next_states = torch.tensor(np.array(next_states))
            actions     = torch.tensor(actions)
            rewards     = torch.tensor(rewards)
            dones       = torch.tensor(dones, dtype=torch.float32)

            # Q-value the network gave the action we actually took
            q_pred = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # best achievable Q from the next state (no grad-it's a target)
            with torch.no_grad():
                # q_next = model(next_states).max(dim=1).values
                q_next = target_model(next_states).max(dim=1).values
                q_target = rewards + gamma * q_next * (1 - dones)   # zero future if done

            """
            # double DQN: the online net picks the next action, the target
            # net evaluates it - decoupling these curbs Q-value overestimation.

            with torch.no_grad():
                next_actions = model(next_states).argmax(dim=1)
                q_next = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                q_target = rewards + gamma * q_next * (1 - dones)   # zero future if done
            """

            loss = criterion(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
        
        state = torch.tensor(next_state)


        screen.fill((0, 0, 0))

        if i % 50 == 0:
        
            rad = CELL_SIZE / 2.0
            #pygame.draw.circle(screen, (255,0,0), (fruit[0] * CELL_SIZE + rad, fruit[1] * CELL_SIZE - rad), rad)
            pygame.draw.rect(screen, (255,0,0), (game.fruit[0] * CELL_SIZE, game.fruit[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            for j in range(game.snake.length):
                if j == 0:
                    color = (255,255,0)
                else:
                    color = (0,255,0)

                x, y = game.snake.positions[j]
                pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            text_surface = game_font.render("Episode: " + str(i), True, (255, 255, 255))
            text_surface2 = game_font.render("Score: " + str(score), True, (255, 255, 255))

            screen.blit(text_surface, (12, 14))
            screen.blit(text_surface2, (12, 28))

            pygame.display.flip()
            clock.tick(1000)

    
    #epsilon = max(epsilon - epsilon_decay, epsilon_min) # linear decay
    epsilon = max(epsilon * epsilon_decay, epsilon_min) # exponential decay
    scores.append(score)
    survival.append(steps)

torch.save(model.state_dict(), 'models/mlp_10x10.pt')
print("Episode " + str(scores.index(max(scores))) + ": " + str(max(scores)))

# ---- Metrics summary (numbers you can cite) ----
final_avg = sum(scores[-100:]) / len(scores[-100:])
final_survival = sum(survival[-100:]) / len(survival[-100:])

# Sample efficiency: first episode whose 50-episode rolling avg reaches half the final avg
target = 0.5 * final_avg
window = 50
episodes_to_target = next(
    (k for k in range(window, len(scores))
    if sum(scores[k - window:k]) / window >= target),
    None,
)

print("\n=== Metrics summary ===")
print(f"Best score:              {max(scores)}")
print(f"Avg score (last 100):    {final_avg:.2f}")
print(f"Avg survival (last 100): {final_survival:.1f} steps")
print(f"Random baseline score:   {baseline_score:.2f}")
if baseline_score > 0:
    print(f"Improvement over random: {final_avg / baseline_score:.1f}x")
else:
    print("Improvement over random: baseline ~0 (random almost never scores)")
print(f"Episodes to reach avg {target:.1f}: {episodes_to_target}")

plt.plot(range(episodes), scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.show()