# Neural Network Architecture

## Overview

The Pong AI uses a feedforward neural network with:
- **Input Layer**: 8 neurons (game state features)
- **Hidden Layer 1**: 64 neurons (ReLU activation)
- **Hidden Layer 2**: 64 neurons (ReLU activation)
- **Output Layer**: 3 neurons (action probabilities)

## Inputs (8 features)

The neural network receives a normalized state vector with 8 values:

### 1. **Paddle Y Position** (normalized)
   - Range: `[0, 1]`
   - Description: Vertical position of the AI's paddle (0 = top, 1 = bottom)
   - Formula: `paddle.rect.centery / HEIGHT`

### 2. **Ball X Position** (normalized)
   - Range: `[0, 1]`
   - Description: Horizontal position of the ball relative to the AI's paddle
   - For left paddle: `ball.centerx / WIDTH` (0 = at paddle, 1 = right edge)
   - For right paddle: `1 - (ball.centerx / WIDTH)` (0 = at paddle, 1 = left edge)

### 3. **Ball Y Position** (normalized)
   - Range: `[0, 1]`
   - Description: Vertical position of the ball (0 = top, 1 = bottom)
   - Formula: `ball.rect.centery / HEIGHT`

### 4. **Ball X Velocity** (normalized)
   - Range: `[-1, 1]`
   - Description: Horizontal velocity of the ball (negative = left, positive = right)
   - Formula: `ball.dx / 25` (normalized by fixed max speed for compatibility)

### 5. **Ball Y Velocity** (normalized)
   - Range: `[-1, 1]`
   - Description: Vertical velocity of the ball (negative = up, positive = down)
   - Formula: `ball.dy / 25` (normalized by fixed max speed for compatibility)

### 6. **Opponent Paddle Y Position** (normalized)
   - Range: `[0, 1]`
   - Description: Vertical position of the opponent's paddle
   - Formula: `opponent.rect.centery / HEIGHT`

### 7. **Distance to Ball** (normalized)
   - Range: `[0, 1]`
   - Description: Euclidean distance from paddle center to ball center
   - Formula: `sqrt((ball_x - paddle_x)² + (ball_y - paddle_y)²) / sqrt(WIDTH² + HEIGHT²)`

### 8. **Ball Approaching** (binary indicator)
   - Range: `[-1, 1]`
   - Description: Whether the ball is moving towards the AI's paddle
   - Value: `1.0` if ball moving towards paddle, `-1.0` if moving away
   - For left paddle: `1.0` if `ball.dx < 0`, else `-1.0`
   - For right paddle: `1.0` if `ball.dx > 0`, else `-1.0`

## Outputs (3 actions)

The neural network outputs 3 values representing action probabilities:

### 0. **Move Up**
   - Action: Move paddle upward
   - Implementation: `paddle.move_up()`

### 1. **Move Down**
   - Action: Move paddle downward
   - Implementation: `paddle.move_down()`

### 2. **Stay** (No movement)
   - Action: Don't move the paddle
   - Implementation: No action taken

## Action Selection

The network uses two modes:

### Stochastic (Training)
```python
probs = softmax(output)
action = sample_from(probs)  # Random sample based on probabilities
```
- Adds exploration during training
- Helps genetic algorithm find diverse strategies

### Deterministic (Testing/Playing)
```python
action = argmax(output)  # Choose highest probability action
```
- Consistent behavior for evaluation
- Used when playing against human

## Network Architecture Details

```
Input (8) → Hidden1 (64, ReLU) → Hidden2 (64, ReLU) → Output (3)
```

- **Total Parameters**: ~4,800 weights and biases
- **Activation**: ReLU for hidden layers, Softmax for output (during action selection)
- **Weight Initialization**: PyTorch default (Kaiming uniform for ReLU)

## Why This Design?

1. **Normalized Inputs**: All inputs are normalized to [0,1] or [-1,1], making the network work across different game speeds
2. **Relative Positions**: Uses relative positions (normalized) rather than absolute pixel coordinates
3. **Velocity Information**: Includes velocity to help predict ball trajectory
4. **Opponent Awareness**: Includes opponent position for strategic play
5. **Distance Metric**: Helps AI judge when to move towards the ball
6. **Approach Indicator**: Binary signal helps AI react to incoming balls

## Example Input Vector

```python
[0.5,    # Paddle at middle height
 0.3,    # Ball 30% across screen from paddle
 0.4,    # Ball at 40% screen height
 0.2,    # Ball moving right at 20% max speed
 -0.1,   # Ball moving up at 10% max speed
 0.6,    # Opponent at 60% screen height
 0.25,   # Ball is 25% of max distance away
 1.0]    # Ball is approaching (moving towards paddle)
```

