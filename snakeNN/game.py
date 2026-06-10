from snake import Snake
import numpy as np
import random
import math


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Maps each direction constant to a (dx, dy) step in grid cells.
# y increases downward (matches pygame and snake.up() doing y -= 1).
DIR_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

# The direction directly behind each heading (a 180-degree reversal).
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

class SnakeGame:
    def __init__(self, height=28, width=28, cell_size=25):
        self.h = height
        self.w = width
        self.cell_size = cell_size
        self.reset()

    def reset(self):
        self.score = 0
        self.snake = Snake(3, (self.w/2, self.h/2))
        self.fruit = self.spawn_fruit()
        self.direction = UP
        self.done = False
        self.frames = 0
        return self.get_state()

    def step(self, action):
        # Prevent 180s: once the snake has a body, it can't reverse straight
        # back onto itself, so ignore the reversal and keep the current heading.
        if self.snake.length > 1 and action == OPPOSITE[self.direction]:
            action = self.direction

        distance1 = math.sqrt((self.snake.positions[0][0] - self.fruit[0]) ** 2 + (self.snake.positions[0][1] - self.fruit[1]) ** 2)

        moves = {
            UP: self.snake.up,
            DOWN: self.snake.down,
            LEFT: self.snake.left,
            RIGHT: self.snake.right,
        }
        moves[action]()

        self.frames += 1
        self.direction = action

        reward = -0.01

        if self._is_collision(self.snake.positions[0]):
            self.done = True
            reward -= 10
            return self.get_state(), reward, self.done, self.score

        x, y = self.snake.positions[0]

        distance2 = math.sqrt((self.snake.positions[0][0] - self.fruit[0]) ** 2 + (self.snake.positions[0][1] - self.fruit[1]) ** 2)
        if distance2 < distance1:
            reward += 0.1
        else:
            reward -= 0.1

        if x == self.fruit[0] and y == self.fruit[1]:
            self.snake.grow()
            self.fruit = self.spawn_fruit()

            reward += 10
            self.score += 1
            self.frames = 0
        
        if self.frames > 100 * self.snake.length:
            self.done = True
            return self.get_state(), reward, self.done, self.score

        return self.get_state(), reward, self.done, self.score

    def spawn_fruit(self):
        fruit = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))
        while fruit in self.snake.positions:
            fruit = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))
        
        return fruit

    def _is_collision(self, point):
        """True if `point` (col, row) hits a wall or the snake's own body."""
        x, y = point
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True
        if point in self.snake.positions[1:]:
            return True
        return False

    def _danger(self):
        """(straight, right, left) danger flags relative to current heading."""
        head = self.snake.positions[0]
        dx, dy = DIR_VECTORS[self.direction]

        straight = (head[0] + dx, head[1] + dy)   # same heading
        right    = (head[0] - dy, head[1] + dx)   # 90 deg clockwise
        left     = (head[0] + dy, head[1] - dx)   # 90 deg counter-clockwise

        return (
            int(self._is_collision(straight)),
            int(self._is_collision(right)),
            int(self._is_collision(left)),
        )

    def get_state(self):

        """
        head = self.snake.positions[0]
        fx, fy = self.fruit

        danger_straight, danger_right, danger_left = self._danger()

        state = [
            # Danger relative to the snake's heading
            danger_straight,
            danger_right,
            danger_left,
            # Current direction (one-hot)
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            # Fruit location relative to the head
            int(fx < head[0]),   # fruit is to the left
            int(fx > head[0]),   # fruit is to the right
            int(fy < head[1]),   # fruit is up
            int(fy > head[1]),   # fruit is down
        ]

        return np.array(state, dtype=np.float32)
        """

        # change to grid state for convolutional

        grid = [[(0,0,0) for _ in range(self.w)] for _ in range(self.h)] # channel 0 = body, ch1 = head, ch2 = fruit

        if self.done:
            return np.array(grid, dtype=np.float32).transpose(2, 0, 1)

        for i in range(len(self.snake.positions)):

            x, y = self.snake.positions[i]

            x = int(x)
            y = int(y)

            if x < 28 and x > 0 and y < 28 and y > 0:
                if i == 0:
                    grid[y][x] = (0, 1, 0) # head
                else:
                    grid[y][x] = (1, 0, 0) # body
        
        fx, fy = self.fruit
        grid[fy][fx] = (0, 0, 1) # fruit

        return np.array(grid, dtype=np.float32).transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
