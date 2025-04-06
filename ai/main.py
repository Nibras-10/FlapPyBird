import os
import argparse

def main():
    """Main function to run Flappy Bird AI"""
    parser = argparse.ArgumentParser(description='Flappy Bird AI - DQN Agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test (default: train)')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes for training (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for experience replay (default: 64)')
    parser.add_argument('--render_every', type=int, default=50,
                        help='Render training every N episodes (default: 50)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode for faster training')
    parser.add_argument('--test_episodes', type=int, default=5,
                        help='Number of episodes for testing (default: 5)')
    parser.add_argument('--model_path', type=str, default='models/flappy_bird_dqn_final.pt',
                        help='Path to the model for testing (default: models/flappy_bird_dqn_final.pt)')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between frames during testing (default: 0.01)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Import training module
        from train import train_agent
        
        # Run training
        print(f"Starting training for {args.episodes} episodes...")
        train_agent(
            episodes=args.episodes,
            batch_size=args.batch_size,
            render_every=args.render_every,
            headless=args.headless
        )
        
    elif args.mode == 'test':
        # Import testing module
        from test import test_agent
        
        # Check if model exists
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Please train a model first.")
            return
        
        # Run testing
        print(f"Testing model from {args.model_path} for {args.test_episodes} episodes...")
        test_agent(
            model_path=args.model_path,
            num_episodes=args.test_episodes,
            delay=args.delay
        )

if __name__ == "__main__":
    main()