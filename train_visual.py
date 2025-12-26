"""
Training script with visualization enabled.
Games are now faster (3 points to win, faster ball/paddle speeds).
"""

from train import train_ai

if __name__ == "__main__":
    print("=" * 60)
    print("Pong AI Training - WITH VISUALIZATION")
    print("=" * 60)
    print("\nGames are now faster:")
    print("  - First to 3 points wins (instead of 11)")
    print("  - Faster ball and paddle speeds")
    print("  - Games typically last 5-15 seconds")
    print("\nPress Ctrl+C to stop training early.\n")
    
    # Start training with visualization
    train_ai(
        num_generations=50,      # Start with 50 generations
        population_size=20,      # Smaller population for faster iteration
        render=True,              # Enable visualization
        save_interval=10         # Save every 10 generations
    )

