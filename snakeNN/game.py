import pygame
import pandas as pd
import numpy as np
from snake import Snake
import random
import torch
from model import SnakeNN


rows = 28
cols = 28
score = 0

CELL_SIZE = 25

pygame.init()
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
clock = pygame.time.Clock()

snake = Snake(1, (rows/2, cols/2), CELL_SIZE)
fruit = (random.randint(0, cols), random.randint(0, rows))

torch.manual_seed(67)
model = SnakeNN()

movement = ''

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and movement != 's':
            if event.key == pygame.K_UP:
                movement = 'u'
            if event.key == pygame.K_DOWN:
                movement = 'd'
            if event.key == pygame.K_RIGHT:
                movement = 'r'
            if event.key == pygame.K_LEFT:
                movement = 'l'

    screen.fill((0, 0, 0))

    match movement:
        case 'u':
            snake.up()
        case 'd':
            snake.down()
        case 'r':
            snake.right()
        case 'l':
            snake.left()

    x, y = snake.positions[0]

    if y < 0 or y > rows * CELL_SIZE:
        movement = 's'
    if x < 0 or x > cols * CELL_SIZE:
        movement = 's'
    if x == fruit[0] and y == fruit[1]:
        score += 1
        fruit = (random.randint(0, cols-1), random.randint(0, rows-1))
    
    rad = CELL_SIZE / 2.0
    #pygame.draw.circle(screen, (255,0,0), (fruit[0] * CELL_SIZE + rad, fruit[1] * CELL_SIZE - rad), rad)
    pygame.draw.rect(screen, (255,0,0), (fruit[0] * CELL_SIZE, fruit[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for i in range(snake.length):
        x, y = snake.positions[i]
        pygame.draw.rect(screen, (0,255,0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)

