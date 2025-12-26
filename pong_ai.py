import pygame
import sys
import random
import numpy as np

# Constants (same as pong.py)
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 100
BALL_SIZE = 15
BALL_SPEED = 5  # Match pong.py for consistent gameplay
PADDLE_SPEED = 6  # Match pong.py for consistent gameplay
FPS = 60
WINNING_SCORE = 7  # First to 7 points wins

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = PADDLE_SPEED
        
    def move_up(self):
        if self.rect.top > 0:
            self.rect.y -= self.speed
            
    def move_down(self):
        if self.rect.bottom < HEIGHT:
            self.rect.y += self.speed
    
    def set_position(self, y):
        """Set paddle position directly (for AI control)"""
        self.rect.y = max(0, min(y, HEIGHT - PADDLE_HEIGHT))
            
    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

class Ball:
    def __init__(self):
        self.speed_multiplier = 1.0  # Speed multiplier that increases after each hit
        self.speed_increase = 0.1  # Increase speed by 10% after each hit
        self.max_speed_multiplier = 3.0  # Cap speed at 3x base speed
        self.reset()
        
    def reset(self):
        self.rect = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.speed_multiplier = 1.0  # Reset speed on new ball
        current_speed = BALL_SPEED * self.speed_multiplier
        self.dx = current_speed * random.choice([-1, 1])
        self.dy = current_speed * random.choice([-1, 1])
        
    def move(self):
        self.rect.x += self.dx
        self.rect.y += self.dy
        
        # Bounce off top and bottom walls
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.dy = -self.dy
            
    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)
        
    def check_collision(self, paddle):
        if self.rect.colliderect(paddle.rect):
            # Increase speed after each hit (standard Pong behavior)
            self.speed_multiplier = min(self.speed_multiplier + self.speed_increase, self.max_speed_multiplier)
            current_speed = BALL_SPEED * self.speed_multiplier
            
            # Reverse horizontal direction
            self.dx = -self.dx
            
            # Adjust angle based on where ball hits paddle (standard Pong physics)
            # relative_y: -1 (top of paddle) to +1 (bottom of paddle)
            relative_y = (paddle.rect.centery - self.rect.centery) / (PADDLE_HEIGHT / 2)
            relative_y = max(-1.0, min(1.0, relative_y))  # Clamp to [-1, 1]
            
            # Calculate new velocity with angle based on hit position
            self.dx = current_speed * (1 if self.dx > 0 else -1)  # Maintain direction
            self.dy = -relative_y * current_speed * 0.8  # Angle based on hit position
            
            # Move ball outside paddle to prevent multiple collisions
            if self.dx > 0:
                self.rect.left = paddle.rect.right
            else:
                self.rect.right = paddle.rect.left
            return True  # Collision occurred
        return False  # No collision

class GameAI:
    """
    Game class modified to support AI players and headless mode for training.
    """
    def __init__(self, render=False, headless=False):
        self.render = render
        self.headless = headless
        
        if not headless:
            # Only initialize display if not already initialized
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pong - AI Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 74)
            self.small_font = pygame.font.Font(None, 36)
        else:
            # Headless mode: Pygame should already be initialized on main thread
            # with SDL_VIDEODRIVER='dummy' to prevent window creation
            # NEVER call pygame.init() here - it's already initialized in app.py
            # Calling it again from a background thread causes macOS crashes
            # Even checking pygame.get_init() might trigger initialization, so just assume it's ready
            self.screen = None
            self.clock = None
        
        # Create paddles
        self.paddle_left = Paddle(10, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.paddle_right = Paddle(WIDTH - 25, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        
        # Create ball
        self.ball = Ball()
        
        # Scores
        self.score_left = 0
        self.score_right = 0
        self.game_over = False
        self.winner = None
        
        # Track game stats for fitness
        self.steps = 0
        self.max_steps = 100000  # Very large limit for web gameplay (about 27 minutes at 60 FPS, effectively no timeout)
        self.hits_left = 0  # Track ball hits for left paddle
        self.hits_right = 0  # Track ball hits for right paddle
        self.missed_left = 0  # Track missed balls (ball passes paddle)
        self.missed_right = 0
        self.idle_frames_left = 0  # Frames where paddle velocity < epsilon
        self.idle_frames_right = 0
        # Ball tracking: only track when ball is approaching, bounded
        self.ball_tracking_distances_left = []  # List of normalized dy when ball approaching
        self.ball_tracking_distances_right = []
        self.stay_actions_left = 0  # Count of stay actions (action in deadzone)
        self.stay_actions_right = 0
        # Edge camping tracking
        self.edge_frames_left = 0  # Frames near top/bottom edge when ball approaching
        self.edge_frames_right = 0
        # Movement variance tracking
        self.paddle_y_history_left = []  # Track paddle y positions for variance calculation
        self.paddle_y_history_right = []
        # Return tracking (ball crosses midline after hit)
        self.returns_left = 0  # Successful returns (ball crosses midline after hit)
        self.returns_right = 0
        self.last_hit_by_left = False  # Track who last hit the ball
        self.last_hit_by_right = False
        self.ball_crossed_midline_after_hit_left = False
        self.ball_crossed_midline_after_hit_right = False
        
        # Status info for display
        self.generation = 0
        self.individual = 0
        self.match = 0
        self.total_individuals = 0
        self.total_matches = 0
        
    def get_state(self, is_left_paddle=True):
        """
        Get game state for AI observation.
        Returns normalized state vector with 10 features:
        [paddle_y, ball_x, ball_y, ball_dx, ball_dy, opponent_paddle_y, 
         signed_dy, signed_dx, time_to_intercept, predicted_intercept_y]
        """
        # Normalize all values to [0, 1] or [-1, 1] range
        if is_left_paddle:
            paddle = self.paddle_left
            opponent = self.paddle_right
            # Normalize ball x relative to left paddle (0 = at paddle, 1 = at right edge)
            ball_x_norm = self.ball.rect.centerx / WIDTH
        else:
            paddle = self.paddle_right
            opponent = self.paddle_left
            # Normalize ball x relative to right paddle (0 = at paddle, 1 = at left edge)
            ball_x_norm = 1 - (self.ball.rect.centerx / WIDTH)
        
        paddle_y_norm = paddle.rect.centery / HEIGHT
        ball_y_norm = self.ball.rect.centery / HEIGHT
        opponent_y_norm = opponent.rect.centery / HEIGHT
        
        # Normalize velocities (using fixed max speed for compatibility across different game speeds)
        # Account for speed multiplier: BALL_SPEED * max_multiplier = 5 * 3.0 = 15
        max_ball_speed = 20  # Increased to handle speed multiplier (5 * 3.0 = 15, with some margin)
        ball_dx_norm = self.ball.dx / max_ball_speed  # Normalize to [-1, 1] range (may slightly exceed)
        ball_dy_norm = self.ball.dy / max_ball_speed
        
        # Signed distances (easier to learn directional movement)
        dx = self.ball.rect.centerx - paddle.rect.centerx
        dy = self.ball.rect.centery - paddle.rect.centery
        signed_dx_norm = dx / WIDTH  # Normalized signed dx in [-1, 1]
        signed_dy_norm = dy / HEIGHT  # Normalized signed dy in [-1, 1]
        
        # Time-to-intercept and predicted intercept position
        # Only calculate if ball is approaching
        if is_left_paddle:
            ball_approaching = self.ball.dx < 0
        else:
            ball_approaching = self.ball.dx > 0
        
        if ball_approaching and abs(self.ball.dx) > 0.1:
            # Calculate time to reach paddle's x position
            if is_left_paddle:
                paddle_x = paddle.rect.centerx
            else:
                paddle_x = paddle.rect.centerx
            
            time_to_intercept = (paddle_x - self.ball.rect.centerx) / self.ball.dx
            time_to_intercept = max(0, time_to_intercept)  # Only positive times
            
            # Predict where ball will be when it reaches paddle
            predicted_y = self.ball.rect.centery + self.ball.dy * time_to_intercept
            
            # Handle wall bounces (simplified: just clamp)
            bounces = 0
            while predicted_y < 0 or predicted_y > HEIGHT:
                if predicted_y < 0:
                    predicted_y = -predicted_y
                    bounces += 1
                elif predicted_y > HEIGHT:
                    predicted_y = 2 * HEIGHT - predicted_y
                    bounces += 1
                if bounces > 5:  # Safety limit
                    predicted_y = np.clip(predicted_y, 0, HEIGHT)
                    break
            
            time_to_intercept_norm = np.clip(time_to_intercept / 5.0, 0, 1)  # Normalize (max ~5 seconds)
            predicted_intercept_y_norm = predicted_y / HEIGHT  # Normalize to [0, 1]
        else:
            time_to_intercept_norm = 0.0
            predicted_intercept_y_norm = 0.5  # Center (neutral value)
        
        state = np.array([
            paddle_y_norm,
            ball_x_norm,
            ball_y_norm,
            ball_dx_norm,
            ball_dy_norm,
            opponent_y_norm,
            signed_dy_norm,  # Signed vertical distance
            signed_dx_norm,  # Signed horizontal distance
            time_to_intercept_norm,  # Time to intercept
            predicted_intercept_y_norm  # Predicted intercept position
        ], dtype=np.float32)
        
        # Mirror state for right paddle to make it symmetric (model always sees "left paddle perspective")
        # This allows the model to work on both sides without retraining
        if not is_left_paddle:
            # Flip horizontal features to mirror the state
            state[3] = -state[3]  # ball_dx_norm: flip sign (ball moving right -> moving left from perspective)
            state[7] = -state[7]  # signed_dx_norm: flip sign (ball to right -> to left from perspective)
            # ball_x_norm is already relative (0 = at paddle, 1 = away), so it's fine
            # All other features are symmetric (y positions, velocities, etc.)
        
        return state
    
    def apply_action(self, action, is_left_paddle=True):
        """
        Apply continuous action to paddle.
        action: float in [-1, 1] where -1 = up, 1 = down, 0 = stay
        Can also accept None or string actions for web compatibility
        """
        paddle = self.paddle_left if is_left_paddle else self.paddle_right
        
        # Handle string actions (for web interface)
        if isinstance(action, str):
            if action == 'up':
                action = -1.0
            elif action == 'down':
                action = 1.0
            else:
                action = 0.0
        elif action is None:
            action = 0.0
        
        # Continuous action: map [-1, 1] to paddle movement
        # action < 0: move up, action > 0: move down
        if action < -0.1:  # Move up (with small deadzone)
            # Scale action to movement amount
            move_amount = abs(action) * paddle.speed
            paddle.rect.y = max(0, paddle.rect.y - move_amount)
        elif action > 0.1:  # Move down (with small deadzone)
            move_amount = abs(action) * paddle.speed
            paddle.rect.y = min(HEIGHT - PADDLE_HEIGHT, paddle.rect.y + move_amount)
        # Otherwise stay (action in [-0.1, 0.1] deadzone)
    
    def step(self, action_left, action_right):
        """
        Run one game step with AI actions.
        Returns: (state_left, state_right, done, info)
        """
        if self.game_over:
            return None, None, True, {}
        
        # Track paddle positions for velocity/idle detection
        prev_y_left = self.paddle_left.rect.centery
        prev_y_right = self.paddle_right.rect.centery
        
        # Apply actions
        self.apply_action(action_left, is_left_paddle=True)
        self.apply_action(action_right, is_left_paddle=False)
        
        # Track idle frames (paddle not moving)
        EPS = 0.5  # Epsilon for "not moving"
        if abs(self.paddle_left.rect.centery - prev_y_left) < EPS:
            self.idle_frames_left += 1
        if abs(self.paddle_right.rect.centery - prev_y_right) < EPS:
            self.idle_frames_right += 1
        
        # Track stay actions (action in deadzone [-0.1, 0.1])
        if isinstance(action_left, (int, float)) and abs(action_left) <= 0.1:
            self.stay_actions_left += 1
        if isinstance(action_right, (int, float)) and abs(action_right) <= 0.1:
            self.stay_actions_right += 1
        
        # Capture ball position before moving (for return tracking)
        prev_ball_x = self.ball.rect.centerx
        
        # Update game
        self.ball.move()
        
        # Track ball hits and returns
        if self.ball.check_collision(self.paddle_left):
            self.hits_left += 1
            self.last_hit_by_left = True
            self.last_hit_by_right = False
            self.ball_crossed_midline_after_hit_left = False
        if self.ball.check_collision(self.paddle_right):
            self.hits_right += 1
            self.last_hit_by_right = True
            self.last_hit_by_left = False
            self.ball_crossed_midline_after_hit_right = False
        
        # Track successful returns (ball crosses midline after hit)
        # A return is successful if: ball was hit by this paddle, then crosses midline
        ball_x = self.ball.rect.centerx
        midline = WIDTH // 2
        
        # Check if ball crossed midline (from left to right for left paddle, right to left for right paddle)
        if self.last_hit_by_left and not self.ball_crossed_midline_after_hit_left:
            # Left paddle hit: ball should cross midline going right
            if prev_ball_x <= midline and ball_x > midline and self.ball.dx > 0:
                self.returns_left += 1
                self.ball_crossed_midline_after_hit_left = True
        
        if self.last_hit_by_right and not self.ball_crossed_midline_after_hit_right:
            # Right paddle hit: ball should cross midline going left
            if prev_ball_x >= midline and ball_x < midline and self.ball.dx < 0:
                self.returns_right += 1
                self.ball_crossed_midline_after_hit_right = True
        
        # Track missed balls (ball passes paddle)
        if self.ball.rect.left <= 0 and self.ball.dx < 0:  # Ball going left past left edge
            self.missed_left += 1
        elif self.ball.rect.right >= WIDTH and self.ball.dx > 0:  # Ball going right past right edge
            self.missed_right += 1
        
        # Ball tracking: only track when ball is approaching, use normalized distance
        # Also track edge camping when ball is approaching
        if self.ball.dx < 0:  # Ball moving toward left paddle
            dy = abs(self.paddle_left.rect.centery - self.ball.rect.centery) / HEIGHT  # Normalized [0, 1]
            self.ball_tracking_distances_left.append(dy)
            # Edge camping: paddle near top or bottom when ball approaching
            paddle_y_norm = self.paddle_left.rect.centery / HEIGHT
            if paddle_y_norm > 0.9 or paddle_y_norm < 0.1:
                self.edge_frames_left += 1
        if self.ball.dx > 0:  # Ball moving toward right paddle
            dy = abs(self.paddle_right.rect.centery - self.ball.rect.centery) / HEIGHT  # Normalized [0, 1]
            self.ball_tracking_distances_right.append(dy)
            # Edge camping: paddle near top or bottom when ball approaching
            paddle_y_norm = self.paddle_right.rect.centery / HEIGHT
            if paddle_y_norm > 0.9 or paddle_y_norm < 0.1:
                self.edge_frames_right += 1
        
        # Track paddle positions for movement variance
        self.paddle_y_history_left.append(self.paddle_left.rect.centery / HEIGHT)
        self.paddle_y_history_right.append(self.paddle_right.rect.centery / HEIGHT)
        
        # Check scoring
        if self.ball.rect.left <= 0:
            self.score_right += 1
            self.ball.reset()
        elif self.ball.rect.right >= WIDTH:
            self.score_left += 1
            self.ball.reset()
            
        # Check winning condition
        if self.score_left >= WINNING_SCORE:
            self.game_over = True
            self.winner = "left"
        elif self.score_right >= WINNING_SCORE:
            self.game_over = True
            self.winner = "right"
        
        self.steps += 1
        if self.steps >= self.max_steps:
            self.game_over = True
            # Winner is whoever has more points
            if self.score_left > self.score_right:
                self.winner = "left"
            elif self.score_right > self.score_left:
                self.winner = "right"
            else:
                self.winner = "tie"
        
        # Get new states
        state_left = self.get_state(is_left_paddle=True)
        state_right = self.get_state(is_left_paddle=False)
        
        info = {
            'score_left': self.score_left,
            'score_right': self.score_right,
            'steps': self.steps
        }
        
        return state_left, state_right, self.game_over, info
    
    def reset(self, seed=None):
        """Reset the game with optional seed for deterministic evaluation"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.score_left = 0
        self.score_right = 0
        self.game_over = False
        self.winner = None
        self.ball.reset()
        self.paddle_left.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.paddle_right.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.steps = 0
        self.hits_left = 0
        self.hits_right = 0
        self.missed_left = 0
        self.missed_right = 0
        self.idle_frames_left = 0
        self.idle_frames_right = 0
        self.ball_tracking_distances_left = []
        self.ball_tracking_distances_right = []
        self.stay_actions_left = 0
        self.stay_actions_right = 0
        self.edge_frames_left = 0
        self.edge_frames_right = 0
        self.paddle_y_history_left = []
        self.paddle_y_history_right = []
        self.returns_left = 0
        self.returns_right = 0
        self.last_hit_by_left = False
        self.last_hit_by_right = False
        self.ball_crossed_midline_after_hit_left = False
        self.ball_crossed_midline_after_hit_right = False
        
        state_left = self.get_state(is_left_paddle=True)
        state_right = self.get_state(is_left_paddle=False)
        return state_left, state_right
    
    def set_status(self, generation=0, individual=0, match=0, total_individuals=0, total_matches=0):
        """Set status information for display"""
        self.generation = generation
        self.individual = individual
        self.match = match
        self.total_individuals = total_individuals
        self.total_matches = total_matches
    
    def draw(self):
        """Draw the game (only if rendering is enabled)"""
        if not self.render or self.headless:
            return
            
        self.screen.fill(BLACK)
        
        # Draw status messages at the top
        if self.generation > 0:
            status_font = pygame.font.Font(None, 28)
            gen_text = status_font.render(f"Generation: {self.generation}", True, WHITE)
            if self.total_individuals > 0:
                ind_text = status_font.render(f"Individual: {self.individual}/{self.total_individuals}", True, WHITE)
            else:
                ind_text = status_font.render(f"Individual: {self.individual}", True, WHITE)
            if self.total_matches > 0:
                match_text = status_font.render(f"Match: {self.match}/{self.total_matches}", True, WHITE)
            else:
                match_text = status_font.render(f"Match: {self.match}", True, WHITE)
            
            # Draw status bar background
            pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, WIDTH, 50))
            pygame.draw.line(self.screen, WHITE, (0, 50), (WIDTH, 50), 1)
            
            # Draw status text
            self.screen.blit(gen_text, (10, 5))
            self.screen.blit(ind_text, (10, 25))
            self.screen.blit(match_text, (WIDTH - 200, 15))
        
        # Draw center line
        pygame.draw.aaline(self.screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
        
        # Draw paddles
        self.paddle_left.draw(self.screen)
        self.paddle_right.draw(self.screen)
        
        # Draw ball
        self.ball.draw(self.screen)
        
        # Draw scores (moved down to make room for status)
        score_y = 70 if self.generation > 0 else 20
        score_left_text = self.font.render(str(self.score_left), True, WHITE)
        score_right_text = self.font.render(str(self.score_right), True, WHITE)
        self.screen.blit(score_left_text, (WIDTH // 4, score_y))
        self.screen.blit(score_right_text, (3 * WIDTH // 4, score_y))
        
        pygame.display.flip()
    
    def run_with_ai(self, ai_left, ai_right, max_frames=None, seed=None):
        """
        Run a game with two AI players.
        Returns fitness scores for both AIs.
        """
        state_left, state_right = self.reset(seed=seed)
        frames = 0
        
        while not self.game_over:
            # Get actions from AIs (handle teacher opponent)
            if hasattr(ai_left, 'get_action_for_side'):
                action_left = ai_left.get_action_for_side(state_left, is_left_paddle=True)
            else:
                action_left = ai_left.get_action(state_left)
            
            if hasattr(ai_right, 'get_action_for_side'):
                action_right = ai_right.get_action_for_side(state_right, is_left_paddle=False)
            else:
                action_right = ai_right.get_action(state_right)
            
            # Step game
            state_left, state_right, done, info = self.step(action_left, action_right)
            
            if self.render and not self.headless:
                self.draw()
                self.clock.tick(FPS)
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            
            frames += 1
            if max_frames and frames >= max_frames:
                break
        
        # Calculate fitness using rescaled formula with anti-camping penalties
        # Primary: +A * (points_scored - points_conceded)
        # Secondary: +B * hits, +R * returns
        # Penalties: -C * missed, -D * idle_frames, -E * steps, -F * ball_tracking, 
        #           -G * stay_actions, -H * edge_camping, -I * low_movement_variance,
        #           -J * no_touch_penalty
        
        A = 75   # Score difference multiplier
        B = 20   # Hit reward (increased from 10)
        R = 30   # Return reward (ball crosses midline after hit)
        C = 150  # Missed penalty (increased from 50)
        D = 0.1  # Idle frames penalty
        E = 0.01 # Time penalty
        F = 100  # Ball tracking penalty coefficient (mean-based, applied once)
        G = 0.02 # Stay actions penalty
        H = 0.2  # Edge camping penalty per frame
        I = 200  # Low movement variance penalty coefficient
        J = 300  # No-touch penalty (hard penalty if never hits ball)
        
        # Calculate ball tracking penalty (mean-based, applied once)
        if len(self.ball_tracking_distances_left) > 0:
            mean_dy_approach_left = np.mean(self.ball_tracking_distances_left)
            tracking_penalty_left = F * mean_dy_approach_left
        else:
            tracking_penalty_left = 0.0
        
        if len(self.ball_tracking_distances_right) > 0:
            mean_dy_approach_right = np.mean(self.ball_tracking_distances_right)
            tracking_penalty_right = F * mean_dy_approach_right
        else:
            tracking_penalty_right = 0.0
        
        # Calculate movement variance penalty (anti-camping)
        if len(self.paddle_y_history_left) > 10:
            std_paddle_y_left = np.std(self.paddle_y_history_left)
            movement_variance_penalty_left = I * max(0, 0.05 - std_paddle_y_left)
        else:
            movement_variance_penalty_left = 0.0
        
        if len(self.paddle_y_history_right) > 10:
            std_paddle_y_right = np.std(self.paddle_y_history_right)
            movement_variance_penalty_right = I * max(0, 0.05 - std_paddle_y_right)
        else:
            movement_variance_penalty_right = 0.0
        
        # Left paddle fitness
        points_diff_left = self.score_left - self.score_right
        fitness_left = A * points_diff_left
        fitness_left += B * self.hits_left
        fitness_left += R * self.returns_left
        fitness_left -= C * self.missed_left
        fitness_left -= D * self.idle_frames_left
        fitness_left -= E * self.steps
        fitness_left -= tracking_penalty_left
        fitness_left -= G * self.stay_actions_left
        fitness_left -= H * self.edge_frames_left
        fitness_left -= movement_variance_penalty_left
        
        # Hard penalty for no-touch games
        if self.hits_left == 0:
            fitness_left -= J
        
        # Right paddle fitness
        points_diff_right = self.score_right - self.score_left
        fitness_right = A * points_diff_right
        fitness_right += B * self.hits_right
        fitness_right += R * self.returns_right
        fitness_right -= C * self.missed_right
        fitness_right -= D * self.idle_frames_right
        fitness_right -= E * self.steps
        fitness_right -= tracking_penalty_right
        fitness_right -= G * self.stay_actions_right
        fitness_right -= H * self.edge_frames_right
        fitness_right -= movement_variance_penalty_right
        
        # Hard penalty for no-touch games
        if self.hits_right == 0:
            fitness_right -= J
        
        # Timeout penalty: if game times out, large penalty unless clearly ahead
        if self.steps >= self.max_steps:
            if self.score_left <= self.score_right:
                fitness_left -= 100  # Large penalty for timeout without clear lead
            if self.score_right <= self.score_left:
                fitness_right -= 100
        
        return fitness_left, fitness_right, info

