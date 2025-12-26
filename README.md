# Pong Game - Two Players

A classic Pong game built with Pygame featuring two-player gameplay.

## Installation

### Setting up a Virtual Environment (Recommended)

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **On Windows**:
     ```bash
     venv\Scripts\activate
     ```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. When you're done, deactivate the virtual environment:
```bash
deactivate
```

### Alternative: Install without Virtual Environment

If you prefer not to use a virtual environment, you can install directly:
```bash
pip install -r requirements.txt
```

## How to Run

Run the game with:
```bash
python pong.py
```

## Controls

- **Left Player**: 
  - `W` - Move paddle up
  - `S` - Move paddle down

- **Right Player**:
  - `↑` (Up Arrow) - Move paddle up
  - `↓` (Down Arrow) - Move paddle down

## Game Features

- Two-player local multiplayer
- Score tracking
- Ball physics with angle-based bounces
- Smooth paddle movement
- 60 FPS gameplay
- AI training with neural networks and genetic algorithms

## 🚀 Web Deployment

Play the game online! The AI is deployed and playable via web browser.

**Deploy to Render:**
1. Push code to GitHub
2. Connect repository to Render
3. Deploy as Web Service
4. See `DEPLOYMENT.md` for detailed instructions

**Local Testing:**
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## AI Training with Genetic Algorithm

This project includes a complete AI training system where two neural networks play against each other and evolve using a genetic algorithm.

### Architecture

- **Neural Network** (`ai_network.py`): PyTorch-based neural network that takes game state as input and outputs paddle actions
- **Game AI** (`pong_ai.py`): Modified game class that supports AI players and headless mode for faster training
- **Genetic Algorithm** (`genetic_algorithm.py`): Implements selection, crossover, and mutation for evolving the population
- **Training Script** (`train.py`): Main training loop that evolves AIs over multiple generations

### Training the AI

1. **Training with visualization (watch the AI learn!)**:
```bash
python train.py --generations 100 --population 50 --render
```
Or use the convenience script:
```bash
python train_visual.py
```
Games are now fast (3 points to win, ~5-15 seconds per game) so you can watch many matches!

2. **Headless training (faster, no visualization)**:
```bash
python train.py --generations 100 --population 50
```

3. **Custom parameters**:
```bash
python train.py --generations 200 --population 100 --render
```

### Testing a Trained Model

After training, test your AI:
```bash
python train.py --test best_model_final.pth
python train.py --test best_model_gen_80.pth
```

### Playing Against the AI

Play against your trained AI in the original game (slower gameplay, 11 points to win):
```bash
python pong_vs_ai.py --model best_model_final.pth
python pong_vs_ai.py --model best_model_gen_80.pth

```

**Note**: Models trained on fast gameplay (3 points, faster speeds) will work perfectly on the slower gameplay (11 points, normal speeds) because:
- State normalization uses fixed maximum values that work across different game speeds
- The neural network learns relative positions and movements, not absolute speeds
- All inputs are normalized to [0,1] or [-1,1] ranges regardless of actual game speed

**Controls**:
- `W` - Move paddle up
- `S` - Move paddle down
- `SPACE` - Restart after game ends

### How It Works

1. **Population**: A population of neural networks (default: 50) is initialized with random weights
2. **Evaluation**: Each AI plays matches against random opponents from the population
3. **Fitness**: Fitness is calculated based on:
   - Score (10 points per goal)
   - Winning bonus (100 points)
   - Rally length bonus (encourages longer games)
4. **Evolution**: 
   - **Elite Selection**: Best performers are kept
   - **Tournament Selection**: Parents are selected via tournament
   - **Crossover**: Offspring combine weights from two parents
   - **Mutation**: Random noise is added to weights
5. **Repeat**: Process continues for specified number of generations

### Model Files

Trained models are saved as `.pth` files containing:
- Network weights
- Fitness score
- Generation number
- Network architecture parameters

### Tips for Training

- **Fast gameplay**: Games now end at 3 points (instead of 11) with faster ball/paddle speeds, so each game lasts only 5-15 seconds
- Start with smaller populations (20-30) for faster initial testing
- Use `--render` to watch the AI learn! Games are fast enough that visualization doesn't slow things down too much
- Training typically takes 30 minutes to several hours depending on parameters
- Models are auto-saved every 10 generations and at the end
- You can interrupt training (Ctrl+C) and the best model will be saved

# neural-net-pong-ai
