

class Snake():
    def __init__(self, length=1, start_pos=(0,0), cell_size=25):
        self.length = length
        self.cell_size = cell_size
        self.positions = []
        x, y = start_pos
        for _ in range(length):
            self.positions.append((x,y))
            y += cell_size
    
    def up(self):
        if self.length > 0:
            x, y = self.positions[0]
            self.positions[0] = (x, y - 1)

    def down(self):
        if self.length > 0:
            x, y = self.positions[0]
            self.positions[0] = (x, y + 1)
    
    def right(self):
        if self.length > 0:
            x, y = self.positions[0]
            self.positions[0] = (x + 1, y)
    
    def left(self):
        if self.length > 0:
            x, y = self.positions[0]
            self.positions[0] = (x - 1, y)
    
    def grow(self):
        x, y = self.positions[-1]
        self.positions.append(())
        self.length += 1