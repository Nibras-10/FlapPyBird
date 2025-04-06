import pygame
import time
import numpy as np
import torch
from game_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent

def test_agent(model_path="models/flappy_bird_dqn_final.pt", num_episodes=10, delay=0.01):
    """
    Test a trained DQN agent on Flappy Bird
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of test episodes to run
        delay: Time delay between frames (smaller is faster)
    """
    # Initialize environment and agent
    env = FlappyBirdWrapper(headless=False)
    state_size = 4
    action_size = 2
    
    # Initialize agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_size, action_size, device)
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    # Performance tracking
    scores = []
    
    # Test loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        # Run episode
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
            
            # Render
            env.render()
            time.sleep(delay)  # Add delay for better visualization
        
        # Record score
        scores.append(info["score"])
        print(f"Episode {episode+1}: Score = {info['score']}, Steps = {steps}")
    
    # Print statistics
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    print(f"\nTest results over {num_episodes} episodes:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Maximum Score: {max_score}")
    print(f"Scores: {scores}")
    
    env.close()

if __name__ == "__main__":
    # Test the agent with the final model
    test_agent("models/flappy_bird_dqn_final.pt", num_episodes=5)