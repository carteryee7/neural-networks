
class Snake():
    def __init__(self, length=1, start_pos=(0, 0)):
        self.length = length
        # Head is positions[0]; the body trails behind it.
        x, y = start_pos
        self.positions = [(x, y + i) for i in range(length)]
        self._grow_pending = 0

    def _move(self, dx, dy):
        # Add a new head one cell over, then drop the tail so the whole
        # body follows. If growth is pending (ate fruit), keep the tail.
        head_x, head_y = self.positions[0]
        self.positions.insert(0, (head_x + dx, head_y + dy))
        if self._grow_pending > 0:
            self._grow_pending -= 1
            self.length += 1
        else:
            self.positions.pop()

    def up(self):
        self._move(0, -1)

    def down(self):
        self._move(0, 1)

    def right(self):
        self._move(1, 0)

    def left(self):
        self._move(-1, 0)

    def grow(self):
        # Defer growth by one cell: the next move keeps the tail instead of
        # popping it, so the body lengthens behind the head.
        self._grow_pending += 1
