import numpy as np
import pygame
import sys
import os

# Add the repository to the path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original game
from src.game import Game
from src.modules.utils import GameMode

class FlappyBirdWrapper:
    """
    Wrapper for the Flappy Bird game to use with RL algorithms.
    This converts the game into an RL environment.
    """
    def __init__(self, headless=False):
        # Initialize game with no sound for training
        self.game = Game()
        self.game.audio.sound_on = not headless
        
        # Game parameters
        self.screen_width = self.game.screen_dims[0]
        self.screen_height = self.game.screen_dims[1]
        
        # For headless mode
        self.headless = headless
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.set_mode((1, 1))
        
        # Initialize game
        self.game.start()
        self.reset()
    
    def reset(self):
        """Reset the game environment"""
        # Restart the game
        self.game.restart()
        
        # Get the initial state
        state = self._get_state()
        return state
    
    def step(self, action):
        """
        Take an action in the environment
        action: 0 (do nothing) or 1 (flap)
        returns: next_state, reward, done, info
        """
        # Default reward for surviving
        reward = 0.1
        
        # Handle events and check if the game is over
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Apply action (0 = do nothing, 1 = flap)
        if action == 1:
            # Simulate a spacebar press
            self.game.input.key_down = True
        else:
            self.game.input.key_down = False
        
        # Update game state
        self.game.update()
        
        # Check if we passed a pipe (scored a point)
        if self.game.score > self._prev_score:
            reward = 1.0
            self._prev_score = self.game.score
        
        # Check if game is over
        if self.game.mode == GameMode.GAME_OVER:
            reward = -10.0
            done = True
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            "score": self.game.score
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Extract state representation for RL agent"""
        bird = self.game.entities["player"]
        pipes = self.game.entities["pipes"]
        
        # Bird position and velocity
        bird_y = bird.pos[1]
        bird_vel = bird.velocity
        
        # Find the next pipe
        next_pipe = None
        for pipe in pipes:
            # Check if pipe is ahead of the bird
            if pipe.pos[0] + pipe.width > bird.pos[0]:
                next_pipe = pipe
                break
        
        # If no pipe ahead, use default values
        if next_pipe is None:
            horizontal_distance = self.screen_width
            gap_y = self.screen_height / 2
        else:
            horizontal_distance = next_pipe.pos[0] - bird.pos[0]
            gap_y = next_pipe.gap_pos
        
        # Calculate vertical distance to the center of the gap
        vertical_distance = bird_y - gap_y
        
        # Normalize values
        normalized_bird_y = bird_y / self.screen_height
        normalized_bird_vel = bird_vel / 10.0  # normalize velocity
        normalized_horizontal_distance = horizontal_distance / self.screen_width
        normalized_vertical_distance = vertical_distance / self.screen_height
        
        return np.array([
            normalized_bird_y,
            normalized_bird_vel,
            normalized_horizontal_distance,
            normalized_vertical_distance
        ])
    
    def render(self):
        """Render the current game state"""
        if not self.headless:
            self.game.render()
    
    def close(self):
        """Close the environment"""
        pygame.quit()