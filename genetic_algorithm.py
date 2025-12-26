import numpy as np
import torch
from ai_network import PongNet
import random

class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving Pong AI neural networks.
    """
    def __init__(self, population_size=50, mutation_rate=0.08, crossover_rate=0.7, 
                 elite_size=3, input_size=10, hidden_size=64):
        self.population_size = population_size
        self.base_mutation_rate = 0.08  # Fixed at 0.08
        self.mutation_rate = 0.08  # Fixed, will only burst on stagnation
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size  # Reduced to 3
        self.input_size = input_size  # Now 10 features
        self.hidden_size = hidden_size
        self.fixed_opponent_indices = None  # Will be set per generation
        
        # Initialize population
        self.population = []
        for _ in range(population_size):
            net = PongNet(input_size=input_size, hidden_size=hidden_size)
            self.population.append(net)
        
        self.fitness_scores = np.zeros(population_size)
        self.generation = 0
        self.best_fitness_history = []  # Track best fitness for adaptive mutation
        self.stagnation_count = 0  # Count generations without improvement
        self.hall_of_fame = []  # Keep best models ever seen (up to 50)
        self.hall_of_fame_size = 50
        self.burst_mode = False  # Track if we're in mutation burst mode
        self.fixed_opponent_indices = None  # Fixed opponent list per generation
    
    def evaluate_population(self, game, matches_per_individual=10, use_fixed_seeds=True, use_teacher=False, visualizer=None):
        """
        Evaluate fitness of all individuals in the population.
        Uses deterministic evaluation with fixed seeds for fairness.
        Curriculum learning: progressive difficulty based on generation.
        Fixed opponent lists per generation for reduced noise.
        """
        from teacher_opponent import TeacherOpponent
        
        self.fitness_scores = np.zeros(self.population_size)
        
        # Fixed seeds for deterministic evaluation (10 seeds)
        fixed_seeds = list(range(1000, 1010))  # 10 fixed seeds
        
        # Curriculum learning: adjust match distribution based on generation
        gen = self.generation
        if gen < 20:
            # Generations 1-20: 8 games vs teacher/weak baseline + 2 vs population
            num_teacher = 8
            num_population = 2
            num_hof = 0
        elif gen < 50:
            # Generations 21-50: 5 vs population + 3 vs HoF + 2 vs teacher
            num_teacher = 2
            num_population = 5
            num_hof = 3
        else:
            # Generations 51+: 5 vs population + 5 vs HoF (teacher off)
            num_teacher = 0
            num_population = 5
            num_hof = 5
        
        # Create teacher if needed by curriculum (regardless of use_teacher flag)
        teacher = TeacherOpponent() if num_teacher > 0 else None
        
        # Create fixed opponent list for this generation (same for all individuals)
        # This reduces ranking noise
        if self.fixed_opponent_indices is None or len(self.fixed_opponent_indices) < num_population:
            # Generate fixed opponent indices for population matches
            # Use generation as seed for reproducibility
            rng = np.random.RandomState(self.generation)
            self.fixed_opponent_indices = []
            for _ in range(num_population):
                idx = rng.randint(0, self.population_size)
                self.fixed_opponent_indices.append(idx)
        
        # Notify visualizer of new generation
        if visualizer:
            visualizer.start_generation(self.generation + 1, self.population_size, len(fixed_seeds))
        
        for i in range(self.population_size):
            fitnesses = []
            match_idx = 0
            
            # Teacher matches
            for _ in range(num_teacher):
                if match_idx >= len(fixed_seeds):
                    break
                if teacher is None:
                    # Safety check: skip if teacher not available
                    match_idx += 1
                    continue
                    
                seed = fixed_seeds[match_idx]
                serve_left = (match_idx % 2 == 0)
                
                game.set_status(
                    generation=self.generation + 1,
                    individual=i + 1,
                    match=match_idx + 1,
                    total_individuals=self.population_size,
                    total_matches=len(fixed_seeds)
                )
                
                class TeacherWrapper:
                    def __init__(self, teacher, is_left):
                        self.teacher = teacher
                        self.is_left = is_left
                    def get_action(self, state):
                        return self.teacher.get_action_for_side(state, is_left_paddle=self.is_left)
                
                if serve_left:
                    teacher_left = TeacherWrapper(teacher, True)
                    fitness, _, _ = game.run_with_ai(
                        self.population[i],
                        teacher_left,
                        max_frames=2000,
                        seed=seed
                    )
                else:
                    teacher_right = TeacherWrapper(teacher, False)
                    _, fitness, _ = game.run_with_ai(
                        teacher_right,
                        self.population[i],
                        max_frames=2000,
                        seed=seed
                    )
                fitnesses.append(fitness)
                match_idx += 1
            
            # Population matches (fixed opponents)
            for pop_match in range(num_population):
                if match_idx >= len(fixed_seeds):
                    break
                seed = fixed_seeds[match_idx]
                serve_left = (match_idx % 2 == 0)
                
                # Use fixed opponent (rotate through list, skip self)
                opponent_idx = self.fixed_opponent_indices[pop_match % len(self.fixed_opponent_indices)]
                if opponent_idx == i:
                    opponent_idx = (opponent_idx + 1) % self.population_size
                
                game.set_status(
                    generation=self.generation + 1,
                    individual=i + 1,
                    match=match_idx + 1,
                    total_individuals=self.population_size,
                    total_matches=len(fixed_seeds)
                )
                
                if serve_left:
                    fitness, _, _ = game.run_with_ai(
                        self.population[i],
                        self.population[opponent_idx],
                        max_frames=2000,
                        seed=seed
                    )
                else:
                    _, fitness, _ = game.run_with_ai(
                        self.population[opponent_idx],
                        self.population[i],
                        max_frames=2000,
                        seed=seed
                    )
                fitnesses.append(fitness)
                match_idx += 1
            
            # Hall-of-Fame matches
            for hof_match in range(num_hof):
                if match_idx >= len(fixed_seeds) or len(self.hall_of_fame) == 0:
                    break
                seed = fixed_seeds[match_idx]
                serve_left = (match_idx % 2 == 0)
                
                hof_idx = hof_match % len(self.hall_of_fame)
                hof_opponent, _ = self.hall_of_fame[hof_idx]
                
                game.set_status(
                    generation=self.generation + 1,
                    individual=i + 1,
                    match=match_idx + 1,
                    total_individuals=self.population_size,
                    total_matches=len(fixed_seeds)
                )
                
                if serve_left:
                    fitness, _, _ = game.run_with_ai(
                        self.population[i],
                        hof_opponent,
                        max_frames=2000,
                        seed=seed
                    )
                else:
                    _, fitness, _ = game.run_with_ai(
                        hof_opponent,
                        self.population[i],
                        max_frames=2000,
                        seed=seed
                    )
                fitnesses.append(fitness)
                match_idx += 1
            
            # Use median fitness (more robust than mean)
            self.fitness_scores[i] = np.median(fitnesses) if fitnesses else 0.0
            
            # Update visualizer
            if visualizer:
                visualizer.update_incremental(i, len(fixed_seeds), self.fitness_scores[i])
    
    def select_parents(self, tournament_size=3):
        """
        Tournament selection: randomly select k individuals and return the best one.
        Reduced tournament size from 5 to 3 to reduce selection pressure.
        """
        tournament_indices = np.random.choice(
            self.population_size, 
            size=tournament_size, 
            replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]
    
    def crossover(self, parent1, parent2, use_neuron_crossover=False):
        """
        Create offspring by combining weights from two parents.
        Options:
        - Blend crossover: child = α*p1 + (1-α)*p2, where α is random in [0.25, 0.75]
        - Neuron-wise crossover: swap whole neurons (preserves structure better)
        """
        child = PongNet(input_size=self.input_size, hidden_size=self.hidden_size)
        
        if use_neuron_crossover:
            # Neuron-wise crossover: swap entire neurons between parents
            child.load_state_dict(parent1.state_dict())
            
            # Crossover hidden layer 1 neurons
            with torch.no_grad():
                # Get parent parameters
                p1_params = list(parent1.parameters())
                p2_params = list(parent2.parameters())
                c_params = list(child.parameters())
                
                # Crossover fc1 (input -> hidden1): swap random neurons
                if len(p1_params) >= 1:
                    fc1_w1, fc1_b1 = p1_params[0].data, p1_params[1].data  # weight, bias
                    fc1_w2, fc1_b2 = p2_params[0].data, p2_params[1].data
                    
                    # Randomly select which neurons to swap (50% chance per neuron)
                    num_neurons = fc1_w1.shape[0]
                    swap_mask = torch.rand(num_neurons) < 0.5
                    
                    c_params[0].data[swap_mask] = fc1_w2[swap_mask]
                    c_params[1].data[swap_mask] = fc1_b2[swap_mask]
                
                # Crossover fc2 (hidden1 -> hidden2): swap random neurons
                if len(p1_params) >= 3:
                    fc2_w1, fc2_b1 = p1_params[2].data, p1_params[3].data
                    fc2_w2, fc2_b2 = p2_params[2].data, p2_params[3].data
                    
                    num_neurons = fc2_w1.shape[0]
                    swap_mask = torch.rand(num_neurons) < 0.5
                    
                    c_params[2].data[swap_mask] = fc2_w2[swap_mask]
                    c_params[3].data[swap_mask] = fc2_b2[swap_mask]
                
                # Crossover fc3 (hidden2 -> output): blend (output layer is small)
                if len(p1_params) >= 5:
                    alpha = np.random.uniform(0.25, 0.75)
                    c_params[4].data = alpha * p1_params[4].data + (1 - alpha) * p2_params[4].data
                    if len(p1_params) > 5:
                        c_params[5].data = alpha * p1_params[5].data + (1 - alpha) * p2_params[5].data
        else:
            # Blend crossover: α in [0.25, 0.75]
            weights1 = parent1.get_weights()
            weights2 = parent2.get_weights()
            alpha = np.random.uniform(0.25, 0.75)
            child_weights = alpha * weights1 + (1 - alpha) * weights2
            child.set_weights(child_weights)
        
        return child
    
    def mutate(self, individual):
        """
        Mutate an individual by adding random noise to weights.
        Fixed mutation: rate=0.08, strength=0.05 normally.
        On stagnation burst: strength=0.10 for one generation, then reset.
        """
        weights = individual.get_weights()
        
        # Fixed mutation strength (0.05 normally, 0.10 in burst mode)
        if self.burst_mode:
            mutation_strength = 0.10  # Burst mode
        else:
            mutation_strength = 0.05  # Normal mode (fixed)
        
        # Fixed mutation rate
        mutation_rate = 0.08
        
        # Add Gaussian noise to random weights
        mutation_mask = np.random.random(size=weights.shape) < mutation_rate
        noise = np.random.normal(0, mutation_strength, size=weights.shape)
        weights[mutation_mask] += noise[mutation_mask]
        
        individual.set_weights(weights)
        return individual
    
    def evolve(self):
        """
        Create next generation using selection, crossover, and mutation.
        Includes adaptive mutation and diversity maintenance.
        """
        # Track fitness for adaptive mutation
        current_best = np.max(self.fitness_scores)
        self.best_fitness_history.append(current_best)
        
        # Check for stagnation (no improvement in last 5 generations)
        if len(self.best_fitness_history) > 5:
            recent_best = max(self.best_fitness_history[-5:])
            if recent_best <= max(self.best_fitness_history[:-5] if len(self.best_fitness_history) > 5 else [0]):
                self.stagnation_count += 1
                # Enter burst mode if stagnating (one generation of high mutation)
                if not self.burst_mode:
                    self.burst_mode = True
            else:
                self.stagnation_count = max(0, self.stagnation_count - 1)
                # Reset burst mode if improving
                self.burst_mode = False
        else:
            self.burst_mode = False
        
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Create new population
        new_population = []
        
        # Update Hall of Fame with best individuals (store copies, not references)
        for i in range(min(5, len(sorted_indices))):  # Add top 5 to consideration
            idx = sorted_indices[i]
            fitness = self.fitness_scores[idx]
            
            # Create a copy of the network (not a reference)
            hof_net = PongNet(input_size=self.input_size, hidden_size=self.hidden_size)
            hof_net.set_weights(self.population[idx].get_weights())
            
            # Add to hall of fame if it's good enough
            if len(self.hall_of_fame) < self.hall_of_fame_size:
                self.hall_of_fame.append((hof_net, fitness))
            else:
                # Replace worst in hall of fame if this is better
                min_hof_fitness = min([f for _, f in self.hall_of_fame])
                if fitness > min_hof_fitness:
                    # Remove worst
                    self.hall_of_fame = [(net, f) for net, f in self.hall_of_fame if f != min_hof_fitness]
                    # Add new one
                    self.hall_of_fame.append((hof_net, fitness))
        
        # Sort hall of fame by fitness
        self.hall_of_fame.sort(key=lambda x: x[1], reverse=True)
        self.hall_of_fame = self.hall_of_fame[:self.hall_of_fame_size]
        
        # Keep elite individuals (best performers) - reduced to 3
        for i in range(self.elite_size):
            elite_net = PongNet(input_size=self.input_size, hidden_size=self.hidden_size)
            elite_net.set_weights(self.population[sorted_indices[i]].get_weights())
            new_population.append(elite_net)
        
        # Add a few random individuals to maintain diversity (1-2 individuals)
        # Reduced from 5% to avoid resetting progress
        num_random = 1 if self.generation < 10 else 2
        for _ in range(num_random):
            random_net = PongNet(input_size=self.input_size, hidden_size=self.hidden_size)
            new_population.append(random_net)
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            
            # Crossover (use neuron-wise crossover for better structure preservation)
            use_neuron_crossover = (self.generation > 10)  # Switch to neuron crossover after initial exploration
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2, use_neuron_crossover=use_neuron_crossover)
            else:
                # No crossover, just copy parent
                child = PongNet(input_size=self.input_size, hidden_size=self.hidden_size)
                child.set_weights(parent1.get_weights())
            
            # Mutate
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Reset fixed opponent indices for next generation
        self.fixed_opponent_indices = None
        
        # Reset burst mode after using it (one generation burst)
        if self.burst_mode:
            self.burst_mode = False
    
    def get_best_individual(self):
        """Get the best performing individual"""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], self.fitness_scores[best_idx]
    
    def get_statistics(self):
        """Get statistics about current generation"""
        return {
            'generation': self.generation,
            'best_fitness': np.max(self.fitness_scores),
            'avg_fitness': np.mean(self.fitness_scores),
            'min_fitness': np.min(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores)
        }
    
    def save_best_model(self, filepath):
        """Save the best model to a file"""
        best_net, best_fitness = self.get_best_individual()
        
        # Save Hall of Fame for resuming training
        hof_data = []
        for hof_net, hof_fitness in self.hall_of_fame:
            hof_data.append((hof_net.state_dict(), hof_fitness))
        
        torch.save({
            'model_state_dict': best_net.state_dict(),
            'fitness': best_fitness,
            'generation': self.generation,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'hall_of_fame': hof_data  # Save HoF for resuming
        }, filepath)
        print(f"Saved best model (fitness: {best_fitness:.2f}, generation: {self.generation}) to {filepath}")
    
    def load_model(self, filepath):
        """Load a model from a file"""
        checkpoint = torch.load(filepath, weights_only=False)
        net = PongNet(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size']
        )
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

