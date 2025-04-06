import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from game_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def train_agent(episodes=2000, batch_size=64, render_every=50, headless=False):
    """Train a DQN agent to play Flappy Bird"""
    # Initialize environment and agent
    env = FlappyBirdWrapper(headless=headless)
    state_size = 4  # Bird height, velocity, horizontal/vertical distance to pipe
    action_size = 2  # Flap or do nothing
    agent = DQNAgent(state_size, action_size, device)
    
    # Load existing model if available
    model_path = "models/flappy_bird_dqn_latest.pt"
    if os.path.exists(model_path):
        agent.load(model_path)
    
    # Training parameters
    update_target_every = 100
    save_weights_every = 50
    
    # Tracking metrics
    scores = []
    epsilon_history = []
    avg_scores = []
    losses = []
    
    start_time = time.time()
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_score = 0
        episode_losses = []
        steps = 0
        
        # For rendering
        should_render = (episode % render_every == 0) and not headless
        
        while not done:
            # Choose and perform action
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_score += reward
            steps += 1
            
            # Render if needed
            if should_render:
                env.render()
            
            # Train the agent
            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss)
        
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()
            print(f"Updated target network at episode {episode}")
        
        # Save model weights periodically
        if episode % save_weights_every == 0:
            agent.save(f"models/flappy_bird_dqn_episode_{episode}.pt")
            agent.save("models/flappy_bird_dqn_latest.pt")
        
        # Track metrics
        scores.append(info["score"])
        epsilon_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        mean_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(mean_loss)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print progress
        print(f"Episode: {episode+1}/{episodes}, Score: {info['score']}, Steps: {steps}, " + 
              f"Epsilon: {agent.epsilon:.4f}, Avg Score (100): {avg_score:.2f}, " + 
              f"Loss: {mean_loss:.6f}, Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Plot progress every 100 episodes
        if episode % 100 == 0 and episode > 0:
            plot_training_progress(scores, avg_scores, epsilon_history, losses, episode)
    
    # Save final model
    agent.save("models/flappy_bird_dqn_final.pt")
    
    # Final plot
    plot_training_progress(scores, avg_scores, epsilon_history, losses, episodes, final=True)
    
    # Close environment
    env.close()
    
    return scores, avg_scores, epsilon_history, losses

def plot_training_progress(scores, avg_scores, epsilon_history, losses, episode, final=False):
    """Plot and save training progress"""
    plt.figure(figsize=(15, 10))
    
    # Plot scores
    plt.subplot(2, 2, 1)
    plt.plot(scores)
    plt.plot(np.arange(len(avg_scores)), avg_scores, 'r')
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Avg Score (100)'])
    
    # Plot epsilon
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_history)
    plt.title('Epsilon History')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(losses)
    plt.title('Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot score distribution
    plt.subplot(2, 2, 4)
    plt.hist(scores, bins=20)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save plot
    if final:
        plt.savefig("plots/training_progress_final.png")
    else:
        plt.savefig(f"plots/training_progress_episode_{episode}.png")
    
    plt.close()

if __name__ == "__main__":
    # Set to True for faster training without visualization
    headless_mode = False
    
    # Train agent
    train_agent(episodes=2000, batch_size=64, render_every=50, headless=headless_mode)