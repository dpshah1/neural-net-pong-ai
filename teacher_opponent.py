"""
Rule-based teacher opponent that moves toward the ball.
Used to provide a baseline for evaluation.
"""
import numpy as np

class TeacherOpponent:
    """
    Simple rule-based opponent that tracks the ball.
    """
    def __init__(self):
        pass
    
    def get_action(self, state):
        """
        Get continuous action based on state.
        state[0] = paddle_y_norm
        state[2] = ball_y_norm
        state[3] = ball_dx_norm (negative = moving toward left, positive = toward right)
        
        Returns: float in [-1, 1] where -1 = up, 1 = down, 0 = stay
        """
        paddle_y = state[0]
        ball_y = state[2]
        ball_dx = state[3]
        
        # Move toward ball if it's approaching
        if ball_dx < -0.1:  # Ball moving toward left (approaching)
            if paddle_y > ball_y + 0.05:  # Paddle below ball
                return -0.8  # Move up (continuous)
            elif paddle_y < ball_y - 0.05:  # Paddle above ball
                return 0.8  # Move down (continuous)
            else:
                return 0.0  # Stay (close enough)
        else:
            # Ball not approaching, move to center
            if paddle_y > 0.55:
                return -0.5  # Move up toward center
            elif paddle_y < 0.45:
                return 0.5  # Move down toward center
            else:
                return 0.0  # Stay at center
    
    def get_action_for_side(self, state, is_left_paddle=True):
        """
        Get continuous action knowing which side we're on.
        Returns: float in [-1, 1]
        """
        paddle_y = state[0]
        ball_y = state[2]
        ball_dx = state[3]
        
        # Determine if ball is approaching this paddle
        if is_left_paddle:
            ball_approaching = ball_dx < -0.1  # Negative dx means moving left
        else:
            ball_approaching = ball_dx > 0.1  # Positive dx means moving right
        
        if ball_approaching:
            # Move toward ball with continuous action
            dy = ball_y - paddle_y  # Positive if ball is below paddle
            # Scale action based on distance (clamped to [-1, 1])
            action = np.clip(dy * 2.0, -1.0, 1.0)
            return action
        else:
            # Ball not approaching, move to center
            center_y = 0.5
            dy = center_y - paddle_y
            action = np.clip(dy * 2.0, -1.0, 1.0)
            return action

