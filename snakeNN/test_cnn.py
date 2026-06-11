import pygame
import torch
from game import SnakeGame
from model import SnakeNN, cnn


cols = 10
rows = 10
CELL_SIZE = 70

pygame.init()
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
clock = pygame.time.Clock()
game_font = pygame.font.Font(None, 20)

runs = int(input("Runs: "))
tick_speed = int(input("Tick Speed: "))
game = SnakeGame(rows, cols, CELL_SIZE)

#model = SnakeNN()

model = cnn
model.load_state_dict(torch.load('models/cnn_model.pt', torch.device("cpu")))
model.eval()


for i in range(runs):
    state = torch.tensor(game.reset()).unsqueeze(0)

    while not game.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.done = True
        
        with torch.no_grad():
            pred = model(state)
            action = pred.argmax().item()
        
        next_state, reward, done, score = game.step(action)

        screen.fill((0, 0, 0))

        text_surface = game_font.render("Run: " + str(i), True, (255, 255, 255))
        text_surface2 = game_font.render("Score: " + str(score), True, (255, 255, 255))

        screen.blit(text_surface, (12, 14))
        screen.blit(text_surface2, (12, 28))
    
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

        pygame.display.flip()
        clock.tick(tick_speed)

        state = torch.tensor(next_state).unsqueeze(0)

