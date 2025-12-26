import pygame
import sys
import numpy as np
from pong_ai import GameAI
from genetic_algorithm import GeneticAlgorithm
from training_visualizer import TrainingVisualizer
import time

def train_ai(num_generations=100, population_size=50, render=False, save_interval=10, resume_from=None, start_generation=0):
    """
    Train Pong AI using genetic algorithm.
    
    Args:
        num_generations: Number of generations to evolve
        population_size: Size of the population
        render: Whether to render games (slower but visual)
        save_interval: Save best model every N generations
        resume_from: Path to model checkpoint to resume from (optional)
        start_generation: Starting generation number (for resuming)
    """
    print("Initializing Genetic Algorithm...")
    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=0.08,  # Fixed at 0.08
        crossover_rate=0.7,
        elite_size=3  # Reduced to 3
    )
    
    # Resume from checkpoint if provided
    if resume_from:
        print(f"\nResuming training from {resume_from}...")
        import torch
        from ai_network import PongNet
        
        checkpoint = torch.load(resume_from, weights_only=False)
        loaded_net = PongNet(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size']
        )
        loaded_net.load_state_dict(checkpoint['model_state_dict'])
        
        # Seed population with the loaded model
        # Replace top 10 individuals with copies of the loaded model (with slight mutations)
        print("Seeding population with loaded model...")
        for i in range(min(10, population_size)):
            ga.population[i].set_weights(loaded_net.get_weights())
            # Add small mutations to create diversity
            if i > 0:  # Keep first one exact
                weights = loaded_net.get_weights()
                mutation_mask = np.random.random(size=weights.shape) < 0.05
                noise = np.random.normal(0, 0.02, size=weights.shape)
                weights[mutation_mask] += noise[mutation_mask]
                ga.population[i].set_weights(weights)
        
        # Load Hall of Fame if available in checkpoint
        if 'hall_of_fame' in checkpoint:
            print(f"Loading Hall of Fame with {len(checkpoint['hall_of_fame'])} members...")
            ga.hall_of_fame = []
            for hof_net_dict, hof_fitness in checkpoint['hall_of_fame']:
                hof_net = PongNet(input_size=checkpoint['input_size'], hidden_size=checkpoint['hidden_size'])
                hof_net.load_state_dict(hof_net_dict)
                ga.hall_of_fame.append((hof_net, hof_fitness))
        
        ga.generation = start_generation
        print(f"Resuming from generation {start_generation + 1}")
        print(f"Loaded model fitness: {checkpoint.get('fitness', 'unknown')}")
    
    print("Initializing Game...")
    # If rendering, don't use headless mode
    game = GameAI(render=render, headless=False if render else True)
    
    print("Initializing Visualizer...")
    visualizer = TrainingVisualizer(max_history=num_generations)
    
    print(f"\nStarting training for {num_generations} generations...")
    print(f"Population size: {population_size}")
    print("-" * 60)
    
    best_fitness_history = []
    avg_fitness_history = []
    
    try:
        for generation in range(start_generation, start_generation + num_generations):
            print(f"\nGeneration {generation + 1} (total: {generation + 1})")
            
            # Update generation in GA
            ga.generation = generation
            
            # Evaluate population
            print("Evaluating population...")
            start_time = time.time()
            # Use 10 fixed seeds, median fitness
            # Curriculum learning handles teacher usage automatically based on generation
            ga.evaluate_population(game, matches_per_individual=10, use_fixed_seeds=True, 
                                  use_teacher=False, visualizer=visualizer)  # Flag ignored, curriculum decides
            eval_time = time.time() - start_time
            
            # Get statistics
            stats = ga.get_statistics()
            best_fitness_history.append(stats['best_fitness'])
            avg_fitness_history.append(stats['avg_fitness'])
            
            # Update visualization
            visualizer.update(generation + 1, stats, ga.fitness_scores)
            
            print(f"  Best fitness: {stats['best_fitness']:.2f}")
            print(f"  Avg fitness: {stats['avg_fitness']:.2f}")
            print(f"  Min fitness: {stats['min_fitness']:.2f}")
            print(f"  Mutation rate: {ga.mutation_rate:.3f} (stagnation: {ga.stagnation_count})")
            print(f"  Evaluation time: {eval_time:.2f}s")
            
            # Save best model periodically
            if (generation + 1) % save_interval == 0:
                ga.save_best_model(f"best_model_gen_{generation + 1}.pth")
            
            # Evolve to next generation
            if generation < num_generations - 1:  # Don't evolve after last generation
                print("Evolving to next generation...")
                ga.evolve()
        
        # Final save
        print("\nTraining complete!")
        ga.save_best_model("best_model_final.pth")
        
        # Save visualization
        visualizer.save("training_progress_final.html")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Training Summary:")
        print(f"  Final best fitness: {best_fitness_history[-1]:.2f}")
        print(f"  Final avg fitness: {avg_fitness_history[-1]:.2f}")
        print(f"  Improvement: {best_fitness_history[-1] - best_fitness_history[0]:.2f}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        ga.save_best_model("best_model_interrupted.pth")
        visualizer.save("training_progress_interrupted.html")
        print("Model and visualization saved.")
    
    finally:
        visualizer.close()
        if not game.headless:
            pygame.quit()
    
    return ga, best_fitness_history, avg_fitness_history

def test_ai(model_path, render=True):
    """
    Test a trained AI by playing against itself.
    """
    from ai_network import PongNet
    import torch
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=False)
    ai = PongNet(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size']
    )
    ai.load_state_dict(checkpoint['model_state_dict'])
    ai.eval()
    
    print("Starting game...")
    game = GameAI(render=render, headless=False)
    
    # Play game with AI vs AI
    fitness_left, fitness_right, info = game.run_with_ai(ai, ai)
    
    print(f"\nGame Over!")
    print(f"Left AI Score: {info['score_left']}")
    print(f"Right AI Score: {info['score_right']}")
    print(f"Winner: {game.winner}")
    
    # Keep window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Pong AI with Genetic Algorithm')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations to train (default: 100)')
    parser.add_argument('--population', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--render', action='store_true',
                        help='Render games during training (slower)')
    parser.add_argument('--test', type=str, default=None,
                        help='Test a trained model (provide model path)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a checkpoint (provide model path)')
    parser.add_argument('--start-gen', type=int, default=0,
                        help='Starting generation number when resuming (default: 0)')
    
    args = parser.parse_args()
    
    if args.test:
        test_ai(args.test, render=True)
    else:
        train_ai(
            num_generations=args.generations,
            population_size=args.population,
            render=args.render,
            resume_from=args.resume,
            start_generation=args.start_gen
        )

