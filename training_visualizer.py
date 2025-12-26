import matplotlib
# Use a backend that doesn't conflict with pygame
# Try different backends in order of preference
# macosx is native macOS backend and should work without conflicts
backend_set = False
selected_backend = None

# Test backends by trying to import them
for backend in ['macosx', 'TkAgg']:
    try:
        # Try to import the backend module to see if it's available
        if backend == 'macosx':
            import importlib
            importlib.import_module('matplotlib.backends.backend_macosx')
        elif backend == 'TkAgg':
            import importlib
            importlib.import_module('matplotlib.backends.backend_tkagg')
        
        matplotlib.use(backend)
        backend_set = True
        selected_backend = backend
        break
    except (ImportError, ValueError, AttributeError, ModuleNotFoundError):
        continue

if not backend_set:
    matplotlib.use('Agg')  # Fallback to non-interactive
    selected_backend = 'Agg'
    print(f"  Warning: No interactive backend available, using Agg (image files only)")

import matplotlib.pyplot as plt
import numpy as np
import threading
import time

class TrainingVisualizer:
    """
    Real-time interactive visualization of training metrics using matplotlib.
    Creates a native window that updates efficiently.
    """
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        self.min_fitness = []
        self.std_fitness = []
        
        # Track current generation evaluation progress
        self.current_generation_fitness = []
        self.current_generation = 0
        self.current_individual = 0
        self.current_match = 0
        self.total_individuals = 0
        self.total_matches = 0
        
        # Create figure with two subplots
        # Try to create figure, fall back to other backends if one fails
        self.interactive_backend = False
        backends_to_try = ['macosx', 'TkAgg']
        
        for backend_name in backends_to_try:
            try:
                matplotlib.use(backend_name, force=True)
                # Clear any existing figures
                plt.close('all')
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
                # If we get here, backend worked
                self.interactive_backend = True
                break
            except (ImportError, RuntimeError, AttributeError, ValueError) as e:
                # Try next backend
                continue
        
        # If all interactive backends failed, use Agg
        if not self.interactive_backend:
            try:
                matplotlib.use('Agg', force=True)
                plt.close('all')
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
                self.image_path = 'training_progress_live.png'
                print("  Using Agg backend (image files only)")
            except Exception as e:
                raise RuntimeError(f"Could not initialize any matplotlib backend: {e}")
        
        self.fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Line chart setup
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.set_title('Fitness Over Generations')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(['Best', 'Average', 'Minimum'], loc='upper left')
        
        # Bar chart setup
        self.ax2.set_xlabel('Fitness')
        self.ax2.set_ylabel('Count')
        self.ax2.set_title('Current Generation Fitness Distribution')
        self.ax2.grid(True, alpha=0.3, axis='y')
        
        # Initialize line plots (empty for now)
        self.line_best, = self.ax1.plot([], [], 'g-', linewidth=2, label='Best', marker='o', markersize=4)
        self.line_avg, = self.ax1.plot([], [], 'b-', linewidth=2, label='Average', marker='s', markersize=4)
        self.line_min, = self.ax1.plot([], [], 'r-', linewidth=2, label='Minimum', marker='^', markersize=4)
        self.area_std = None
        
        # Bar chart elements
        self.bars = None
        self.vlines = []
        self.vline_labels = []
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')
        
        # Thread-safe update flag
        self.update_needed = False
        self.lock = threading.Lock()
        
        # Initialize window creation - will be done after pygame initializes
        self.window_created = False
        if not hasattr(self, 'image_path'):
            self.image_path = 'training_progress_live.png'
        
        # Only try to create window if we have an interactive backend
        if hasattr(self, 'interactive_backend') and self.interactive_backend:
            # Don't create window immediately - wait for first update
            # This avoids conflicts with pygame initialization
            print("  Training visualization window will open after initialization...")
        else:
            print("  Training visualization will be saved to: training_progress_live.png")
            print("  The image will update automatically as training progresses.")
    
    def _ensure_window_created(self):
        """Create the window if it hasn't been created yet (called from main thread)"""
        if self.window_created or not hasattr(self, 'interactive_backend') or not self.interactive_backend:
            return
        
        try:
            plt.ion()  # Turn on interactive mode
            # Show the figure - macosx backend doesn't support block parameter
            try:
                self.fig.show(block=False)
            except TypeError:
                # Some backends don't support block parameter
                self.fig.show()
            plt.pause(0.01)  # Small pause to ensure window is created
            self.window_created = True
            if not hasattr(self, '_window_announced'):
                print("  Training visualization window opened!")
                print("  The window will update automatically as training progresses.")
                self._window_announced = True
        except Exception as e:
            # If window creation fails, fall back to saving images
            if not hasattr(self, '_window_failed'):
                print(f"  Warning: Could not create visualization window: {e}")
                print("  Falling back to saving images to training_progress_live.png")
                self._window_failed = True
            self.window_created = False
    
    def _update_plot(self):
        """Update the plots with current data"""
        # Ensure window is created on first update (after pygame has initialized)
        if hasattr(self, 'interactive_backend') and self.interactive_backend and not self.window_created:
            self._ensure_window_created()
        
        with self.lock:
            # Update status text
            status = f"Generation {self.current_generation}"
            if self.total_individuals > 0:
                status += f" | Individual: {self.current_individual}/{self.total_individuals}"
            if self.total_matches > 0:
                status += f" | Match: {self.current_match}/{self.total_matches}"
            self.status_text.set_text(status)
            
            # Update line chart
            if len(self.generations) > 0:
                self.line_best.set_data(self.generations, self.best_fitness)
                self.line_avg.set_data(self.generations, self.avg_fitness)
                self.line_min.set_data(self.generations, self.min_fitness)
                
                # Update std dev area
                if len(self.avg_fitness) > 0 and len(self.std_fitness) > 0:
                    if self.area_std is not None:
                        self.area_std.remove()
                    
                    upper = np.array(self.avg_fitness) + np.array(self.std_fitness)
                    lower = np.array(self.avg_fitness) - np.array(self.std_fitness)
                    self.area_std = self.ax1.fill_between(
                        self.generations, upper, lower,
                        alpha=0.2, color='blue', label='±1 Std Dev'
                    )
                
                # Update axes limits
                all_fitness = self.best_fitness + self.avg_fitness + self.min_fitness
                if len(all_fitness) > 0:
                    y_min = min(all_fitness)
                    y_max = max(all_fitness)
                    y_padding = (y_max - y_min) * 0.1 or 1
                    self.ax1.set_xlim(min(self.generations) - 0.5, max(self.generations) + 0.5)
                    self.ax1.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Update bar chart
            if len(self.current_generation_fitness) > 0:
                # Clear old bars and vlines
                if self.bars is not None:
                    for bar in self.bars:
                        bar.remove()
                for vline in self.vlines:
                    vline.remove()
                for label in self.vline_labels:
                    label.remove()
                self.bars = None
                self.vlines = []
                self.vline_labels = []
                
                # Calculate histogram
                min_fit = min(self.current_generation_fitness)
                max_fit = max(self.current_generation_fitness)
                num_bins = min(20, max(5, len(self.current_generation_fitness) // 2))
                
                counts, bins = np.histogram(self.current_generation_fitness, bins=num_bins)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_width = bins[1] - bins[0]
                
                # Color bars based on fitness (red to yellow to green)
                colors = []
                for center in bin_centers:
                    normalized = (center - min_fit) / (max_fit - min_fit + 1e-6)
                    normalized = max(0.0, min(1.0, normalized))
                    if normalized < 0.5:
                        r, g, b = 1.0, normalized * 2, 0.0
                    else:
                        r, g, b = (1 - normalized) * 2, 1.0, 0.0
                    colors.append((r, g, b))
                
                # Draw bars
                self.bars = self.ax2.bar(bin_centers, counts, width=bin_width * 0.8, 
                                         color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add vertical lines for statistics
                current_best = max(self.current_generation_fitness)
                current_avg = np.mean(self.current_generation_fitness)
                current_min = min(self.current_generation_fitness)
                
                stats = [
                    (current_best, 'green', 'Best'),
                    (current_avg, 'blue', 'Avg'),
                    (current_min, 'red', 'Min')
                ]
                
                y_max_count = max(counts) if len(counts) > 0 else 1
                for i, (val, color, label) in enumerate(stats):
                    vline = self.ax2.axvline(val, color=color, linestyle='--', linewidth=2, alpha=0.8)
                    self.vlines.append(vline)
                    
                    text = self.ax2.text(val, y_max_count * (0.95 - i * 0.1), 
                                        f'{label}: {val:.1f}',
                                        color=color, fontsize=9, ha='center',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    self.vline_labels.append(text)
                
                # Update bar chart axes
                self.ax2.set_xlim(min_fit - bin_width, max_fit + bin_width)
                self.ax2.set_ylim(0, y_max_count * 1.1)
            else:
                # Clear bar chart if no data
                self.ax2.clear()
                self.ax2.set_xlabel('Fitness')
                self.ax2.set_ylabel('Count')
                self.ax2.set_title('Current Generation Fitness Distribution')
                self.ax2.grid(True, alpha=0.3, axis='y')
            
            # Update the window if it exists, otherwise save to file
            if self.window_created:
                try:
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                except:
                    # Window might have been closed, fall back to saving
                    self.window_created = False
                    self.image_path = 'training_progress_live.png'
                    self.fig.savefig(self.image_path, dpi=100, bbox_inches='tight')
            else:
                # Save to file if window not available
                if not hasattr(self, 'image_path'):
                    self.image_path = 'training_progress_live.png'
                self.fig.savefig(self.image_path, dpi=100, bbox_inches='tight')
    
    def start_generation(self, generation, total_individuals, total_matches):
        """Clear bar chart and prepare for new generation"""
        with self.lock:
            self.current_generation = generation
            self.current_generation_fitness = []
            self.total_individuals = total_individuals
            self.total_matches = total_matches
            self.current_individual = 0
            self.current_match = 0
        self._update_plot()
    
    def update_incremental(self, individual_idx, match_idx, fitness_score):
        """Update bar chart with median fitness for each individual (called once per individual)"""
        with self.lock:
            self.current_individual = individual_idx + 1
            self.current_match = match_idx  # This is now the total number of matches
            # Add the median fitness score (this is already the median from all matches)
            self.current_generation_fitness.append(float(fitness_score))
        # Update plot after each individual (since we only get called once per individual now)
        self._update_plot()
    
    def update(self, generation, stats, current_fitness_scores=None):
        """
        Update plots with new generation data.
        
        Args:
            generation: Current generation number
            stats: Dictionary with 'best_fitness', 'avg_fitness', 'min_fitness', 'std_fitness'
            current_fitness_scores: Array of fitness scores for current generation (for bar chart)
        """
        with self.lock:
            # Add to history
            self.generations.append(generation)
            self.best_fitness.append(float(stats['best_fitness']))
            self.avg_fitness.append(float(stats['avg_fitness']))
            self.min_fitness.append(float(stats['min_fitness']))
            self.std_fitness.append(float(stats.get('std_fitness', 0)))
            
            # Keep only last max_history points
            if len(self.generations) > self.max_history:
                self.generations = self.generations[-self.max_history:]
                self.best_fitness = self.best_fitness[-self.max_history:]
                self.avg_fitness = self.avg_fitness[-self.max_history:]
                self.min_fitness = self.min_fitness[-self.max_history:]
                self.std_fitness = self.std_fitness[-self.max_history:]
        
        self._update_plot()
    
    def save(self, filepath='training_progress_final.png'):
        """Save the current plot to a file"""
        self._update_plot()  # Ensure latest data is shown
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {filepath}")
    
    def close(self):
        """Close the plot window"""
        plt.close(self.fig)
