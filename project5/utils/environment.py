import numpy as np
import random

class MouseEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        
        # Elements on the grid (from mouse.py)
        self.EMPTY = 0
        self.MOUSE = 1
        self.CHEESE = 2
        self.TRAP = 3
        self.WALL = 4
        self.ORGANIC_CHEESE = 5
        
        # Action space
        self.ACTIONS = ['up', 'down', 'left', 'right']
        self.ACTION_TO_DELTA = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }
        
        # Counts
        self.NUM_TRAPS = 2
        self.NUM_WALLS = 2
        self.NUM_ORGANIC_CHEESE = 1
        self.NUM_CHEESE = 2
        
        self.reset()
    
    def reset(self):
        """Initialize a new episode"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.mouse_pos = None
        self.cheese_positions = []
        self.organic_cheese_positions = []
        self.trap_positions = []
        self.wall_positions = []
        
        self._place_elements()
        return self._get_state()
    
    def _place_elements(self):
        """Place all elements on the grid using logic from mouse.py"""
        # Place mouse
        while True:
            self.mouse_pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if self.grid[self.mouse_pos] == self.EMPTY:
                self.grid[self.mouse_pos] = self.MOUSE
                break
        
        # Place normal cheese
        for _ in range(self.NUM_CHEESE):
            while True:
                cheese_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[cheese_pos] == self.EMPTY:
                    self.grid[cheese_pos] = self.CHEESE
                    self.cheese_positions.append(cheese_pos)
                    break
        
        # Place organic cheese
        for _ in range(self.NUM_ORGANIC_CHEESE):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[pos] == self.EMPTY:
                    self.grid[pos] = self.ORGANIC_CHEESE
                    self.organic_cheese_positions.append(pos)
                    break
        
        # Place traps
        for _ in range(self.NUM_TRAPS):
            while True:
                trap_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[trap_pos] == self.EMPTY:
                    self.grid[trap_pos] = self.TRAP
                    self.trap_positions.append(trap_pos)
                    break
        
        # Place walls
        for _ in range(self.NUM_WALLS):
            while True:
                wall_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[wall_pos] == self.EMPTY:
                    self.grid[wall_pos] = self.WALL
                    self.wall_positions.append(wall_pos)
                    break
    
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        old_pos = self.mouse_pos
        delta = self.ACTION_TO_DELTA[self.ACTIONS[action]]
        new_pos = (self.mouse_pos[0] + delta[0], self.mouse_pos[1] + delta[1])
        
        # Check bounds and walls
        if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size and 
            self.grid[new_pos] != self.WALL):
            
            # Clear old position
            self.grid[self.mouse_pos] = self.EMPTY
            
            # Calculate reward based on what's at new position
            reward = self._get_reward(new_pos)
            
            # Update mouse position
            self.mouse_pos = new_pos
            
            # Check if collected cheese/organic cheese
            if self.grid[new_pos] in [self.CHEESE, self.ORGANIC_CHEESE]:
                if new_pos in self.cheese_positions:
                    self.cheese_positions.remove(new_pos)
                if new_pos in self.organic_cheese_positions:
                    self.organic_cheese_positions.remove(new_pos)
            
            # Place mouse at new position
            self.grid[self.mouse_pos] = self.MOUSE
            
        else:
            # Invalid move - small penalty
            reward = -0.5
        
        # Episode ends if all cheese collected or hit trap
        done = (len(self.cheese_positions) + len(self.organic_cheese_positions) == 0 or 
                reward == -50)
        
        return self._get_state(), reward, done, {}
    
    def _get_reward(self, pos):
        """Calculate reward based on position (matching your specification)"""
        if self.grid[pos] == self.CHEESE:
            return 10  # Regular cheese gives +10 reward
        elif self.grid[pos] == self.ORGANIC_CHEESE:
            return 10  # Organic cheese also gives +10 reward (same as regular - RLHF will teach preference)
        elif self.grid[pos] == self.TRAP:
            return -50  # Trap gives -50 penalty
        else:
            return -0.2  # Empty cell or wall bump gives -0.2 penalty
    
    def _get_state(self):
        """Return current grid state"""
        return self.grid.copy()
    
    def render(self):
        """Print current grid state"""
        symbols = {
            self.EMPTY: '.',
            self.MOUSE: 'M',
            self.CHEESE: 'C',
            self.TRAP: 'T',
            self.WALL: '#',
            self.ORGANIC_CHEESE: 'O'
        }
        for row in self.grid:
            print(' '.join(symbols[cell] for cell in row))
        print()
    
    def get_action_space(self):
        """Return number of possible actions"""
        return len(self.ACTIONS)
                
        # Place normal cheese
        for _ in range(self.NUM_CHEESE):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[pos] == self.EMPTY:
                    self.grid[pos] = self.CHEESE
                    self.cheese_positions.append(pos)
                    break
                    
        # Place organic cheese
        for _ in range(self.NUM_ORGANIC_CHEESE):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[pos] == self.EMPTY:
                    self.grid[pos] = self.ORGANIC_CHEESE
                    self.organic_cheese_positions.append(pos)
                    break
                    
        # Place traps
        for _ in range(self.NUM_TRAPS):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[pos] == self.EMPTY:
                    self.grid[pos] = self.TRAP
                    self.trap_positions.append(pos)
                    break
                    
        # Place walls
        for _ in range(self.NUM_WALLS):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[pos] == self.EMPTY:
                    self.grid[pos] = self.WALL
                    self.wall_positions.append(pos)
                    break
                    
    def _get_state(self):
        # Create a one-hot encoded version of the grid
        state = np.zeros((6, self.grid_size, self.grid_size))  # 6 channels for each element type
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                element = self.grid[i, j]
                if element == self.EMPTY:
                    state[0, i, j] = 1
                elif element == self.MOUSE:
                    state[1, i, j] = 1
                elif element == self.CHEESE:
                    state[2, i, j] = 1
                elif element == self.TRAP:
                    state[3, i, j] = 1
                elif element == self.WALL:
                    state[4, i, j] = 1
                elif element == self.ORGANIC_CHEESE:
                    state[5, i, j] = 1
                    
        return state
    
    def step(self, action):
        # Convert integer action to string if needed
        if isinstance(action, int):
            action = self.ACTIONS[action]
            
        delta = self.ACTION_TO_DELTA[action]
        new_pos = (self.mouse_pos[0] + delta[0], self.mouse_pos[1] + delta[1])
        
        # Check bounds
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size and 
            self.grid[new_pos] != self.WALL):
            
            # Get reward
            reward = self._get_reward(new_pos)
            
            # Move mouse
            self.grid[self.mouse_pos] = self.EMPTY
            self.grid[new_pos] = self.MOUSE
            self.mouse_pos = new_pos
            
            # Check if cheese was collected
            if new_pos in self.cheese_positions:
                self.cheese_positions.remove(new_pos)
            elif new_pos in self.organic_cheese_positions:
                self.organic_cheese_positions.remove(new_pos)
                
            done = len(self.cheese_positions) + len(self.organic_cheese_positions) == 0
            return self._get_state(), reward, done, {}
        else:
            # Invalid move (hit wall or out of bounds)
            return self._get_state(), -0.2, False, {}
            
    def _get_reward(self, pos):
        if self.grid[pos] == self.CHEESE or self.grid[pos] == self.ORGANIC_CHEESE:
            return 10
        elif self.grid[pos] == self.TRAP:
            return -50
        else:
            return -0.2
            
    def render(self):
        symbols = {
            self.EMPTY: '.',
            self.MOUSE: 'M',
            self.CHEESE: 'C',
            self.TRAP: 'T',
            self.WALL: '#',
            self.ORGANIC_CHEESE: 'O'
        }
        for row in self.grid:
            print(' '.join(symbols[cell] for cell in row))
        print()