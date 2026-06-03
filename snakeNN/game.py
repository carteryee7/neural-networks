from snake import Snake
import numpy as np
import random


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Maps each direction constant to a (dx, dy) step in grid cells.
# y increases downward (matches pygame and snake.up() doing y -= 1).
DIR_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

class SnakeGame:
    def __init__(self, height=28, width=28, cell_size=25):
        self.h = height
        self.w = width
        self.cell_size = cell_size
        self.reset()

    def reset(self):
        self.score = 0
        self.snake = Snake(1, (self.h/2, self.w/2))
        self.fruit = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))
        self.direction = UP
        self.done = False
        return self.get_state()

    def step(self, action):
        match action:
            case int(UP):
                self.snake.up()
            case int(DOWN):
                self.snake.down()
            case int(LEFT):
                self.snake.left()
            case int(RIGHT):
                self.snake.right()
        
        self.direction = action

        reward = -0.01

        if self._is_collision(self.snake.positions[0]):
            self.done = True
            reward -= 10
            return self.get_state(), reward, self.done, self.score

        x, y = self.snake.positions[0]
        if x == self.fruit[0] and y == self.fruit[1]:
            self.snake.grow()
            self.fruit = (random.randint(0, self.w - 1), random.randint(0, self.h - 1))

            reward += 10
            self.score += 1
        
        return self.get_state(), reward, self.done, self.score
        

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