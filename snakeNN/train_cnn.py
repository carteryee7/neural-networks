import pygame
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from game import SnakeGame
from model import SnakeNN, cnn
from collections import deque
import copy
import random
import matplotlib.pyplot as plt

rows = 8
cols = 8

CELL_SIZE = 50

pygame.init()
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
clock = pygame.time.Clock()
game_font = pygame.font.Font(None, 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)

torch.manual_seed(67)
#model = SnakeNN()
model = cnn
model.to(device)

target_model = copy.deepcopy(model)
target_model.to(device)
target_model.eval()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

game = SnakeGame(rows, cols, CELL_SIZE, state_mode="grid")
episodes = 1000
gamma = 0.9
epsilon = 1.0
epsilon_decay = .99
epsilon_min = .01

memory = deque(maxlen=100_000)
batch_size = 64
target_update_freq = 1000
step_count = 0
scores = []

for i in range(episodes):

    state = torch.tensor(game.reset(), dtype=torch.float32, device=device).unsqueeze(0)

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

        memory.append((state.detach().cpu(), action, reward, next_state, done))

        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)
            states      = torch.cat(states, dim=0).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            actions     = torch.tensor(actions, dtype=torch.long, device=device)
            rewards     = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones       = torch.tensor(dones, dtype=torch.float32, device=device)

            # Q-value the network gave the action we actually took
            q_pred = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Best achievable Q from the next state (no grad — it's a target)
            with torch.no_grad():
                q_next = target_model(next_states).max(dim=1).values
                q_target = rewards + gamma * q_next * (1 - dones)   # zero future if done

            loss = criterion(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
        
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)


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

torch.save(model.state_dict(), 'models/cnn_8x8.pt')
print("Episode " + str(scores.index(max(scores))) + ": " + str(max(scores)))

plt.plot(range(episodes), scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.show()