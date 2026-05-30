import pygame
import numpy as np
import pandas as pd
import random

pygame.init()
screen = pygame.display.set_mode((700, 700))
clock = pygame.time.Clock()

rows = 28
cols = 28

CELL_SIZE = 25

#grid = [[0 for _ in range(rows)] for _ in range(cols)]
csv = 'train.csv'
my_df = pd.read_csv(csv)

def func(x):
    if x > 20:
        return 1
    else:
        return 0

vectorized_func = np.vectorize(func)
# Train Test Split!  Set X, y
x = pd.DataFrame(vectorized_func(my_df.drop('label', axis=1)))
grids = np.array(x.loc[:50])
i = 0
grid = grids[i].reshape(28,28)

# data visualizer (list of grids)
# group by number
# scroll through entries

clear = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                running = False
            if event.key == pygame.K_RIGHT:
                i += 1
                clear = 0
            if event.key == pygame.K_LEFT:
                i -= 1
                clear = 0
            if event.key == pygame.K_r:
                clear = 1

    if clear == 1:
        grid = np.array([[0 for _ in range(rows)] for _ in range(cols)])
        clear = 2
    elif clear == 0:
        grid = grids[i].reshape(28,28)

    mouse_buttons = pygame.mouse.get_pressed()
    
    if mouse_buttons[0]:
        # This code runs every frame the left button is held down
        x, y = pygame.mouse.get_pos()
        grid[int(y / CELL_SIZE)][int(x / CELL_SIZE)] = 1
        grid[int(y / CELL_SIZE) - 1][int(x / CELL_SIZE) - 1] = 1
        grid[int(y / CELL_SIZE) + 1][int(x / CELL_SIZE) + 1] = 1
        grid[int(y / CELL_SIZE) + 1][int(x / CELL_SIZE) - 1] = 1
        grid[int(y / CELL_SIZE) - 1][int(x / CELL_SIZE) + 1] = 1
        grid[int(y / CELL_SIZE) - 1][int(x / CELL_SIZE)] = 1
        grid[int(y / CELL_SIZE) + 1][int(x / CELL_SIZE)] = 1
        grid[int(y / CELL_SIZE)][int(x / CELL_SIZE + 1)] = 1
        grid[int(y / CELL_SIZE)][int(x / CELL_SIZE - 1)] = 1
    
    screen.fill((0, 0, 0))

    for x in range(rows):
        for y in range(cols):
            r = 255 * grid[x, y]
            pygame.draw.rect(screen, (r, r, r), (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(60)

arr = np.array(grid)
row = arr.flatten()

#num = int(input("number: "))
#row = np.concatenate((np.array([num]), row))