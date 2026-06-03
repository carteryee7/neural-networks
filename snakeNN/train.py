import pygame
import pandas as pd
import numpy as np
import torch
from game import SnakeGame
from model import SnakeNN

rows = 28
cols = 28

CELL_SIZE = 25

pygame.init()
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
clock = pygame.time.Clock()

torch.manual_seed(67)
model = SnakeNN()

episodes = 50

for i in range(episodes):

    game = SnakeGame()

    while not game.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.done = True

        state = game.get_state()
        action = model(state)
        
        output = game.step(action)
        
        screen.fill((0, 0, 0))
        
        rad = CELL_SIZE / 2.0
        #pygame.draw.circle(screen, (255,0,0), (fruit[0] * CELL_SIZE + rad, fruit[1] * CELL_SIZE - rad), rad)
        pygame.draw.rect(screen, (255,0,0), (game.fruit[0] * CELL_SIZE, game.fruit[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for i in range(game.snake.length):
            x, y = game.snake.positions[i]
            pygame.draw.rect(screen, (0,255,0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        clock.tick(10)




