"""
Quick example script to start training the AI.
This is a simplified version - use train.py for full control.
"""

from train import train_ai

if __name__ == "__main__":
    print("=" * 60)
    print("Pong AI Training - Quick Start")
    print("=" * 60)
    print("\nThis will train AIs using genetic algorithm.")
    print("Training will run headless (no visualization) for speed.")
    print("Press Ctrl+C to stop early (model will be saved).\n")
    
    # Start training with reasonable defaults
    train_ai(
        num_generations=50,      # Start with 50 generations
        population_size=30,      # Smaller population for faster training
        render=False,            # Headless mode (faster)
        save_interval=10         # Save every 10 generations
    )

