import pygame
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from game import SnakeGame
from model import SnakeNN
from collections import deque
import random
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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

game = SnakeGame(rows, cols, CELL_SIZE)
episodes = 1000
gamma = 0.9
epsilon = 1.0
epsilon_decay = .99
epsilon_min = .01

memory = deque(maxlen=100_000)
batch_size = 64
scores = []

for i in range(episodes):

    state = torch.tensor(game.reset())

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

            # Best achievable Q from the next state (no grad — it's a target)
            with torch.no_grad():
                q_next = model(next_states).max(dim=1).values
                q_target = rewards + gamma * q_next * (1 - dones)   # zero future if done

            loss = criterion(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
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

torch.save(model.state_dict(), 'model_10x10.pt')
print("Episode " + str(scores.index(max(scores))) + ": " + str(max(scores)))

plt.plot(range(episodes), scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.show()