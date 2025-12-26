import pygame
import sys
import random
import numpy as np
import torch
from ai_network import PongNet

# Initialize Pygame
pygame.init()

# Constants (original game speed)
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 100
BALL_SIZE = 15
BALL_SPEED = 5
PADDLE_SPEED = 6
FPS = 60
WINNING_SCORE = 11

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

class GameVsAI:
    def __init__(self, ai_model_path):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong - You vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)
        
        # Load AI
        print(f"Loading AI from {ai_model_path}...")
        checkpoint = torch.load(ai_model_path, weights_only=False)
        self.ai = PongNet(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size']
        )
        self.ai.load_state_dict(checkpoint['model_state_dict'])
        self.ai.eval()
        print("AI loaded successfully!")
        
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
        
    def get_state_for_ai(self):
        """
        Get game state for AI observation (same normalization as training).
        Returns normalized state vector with 10 features for the right paddle (AI).
        """
        paddle = self.paddle_right
        opponent = self.paddle_left
        
        # Normalize ball x relative to right paddle (0 = at paddle, 1 = at left edge)
        ball_x_norm = 1 - (self.ball.rect.centerx / WIDTH)
        
        paddle_y_norm = paddle.rect.centery / HEIGHT
        ball_y_norm = self.ball.rect.centery / HEIGHT
        opponent_y_norm = opponent.rect.centery / HEIGHT
        
        # Normalize velocities (using fixed max speed for compatibility)
        # Account for speed multiplier: BALL_SPEED * max_multiplier = 5 * 3.0 = 15
        max_ball_speed = 20  # Increased to handle speed multiplier (5 * 3.0 = 15, with some margin)
        ball_dx_norm = self.ball.dx / max_ball_speed
        ball_dy_norm = self.ball.dy / max_ball_speed
        
        # Signed distances (easier to learn directional movement)
        dx = self.ball.rect.centerx - paddle.rect.centerx
        dy = self.ball.rect.centery - paddle.rect.centery
        signed_dx_norm = dx / WIDTH  # Normalized signed dx in [-1, 1]
        signed_dy_norm = dy / HEIGHT  # Normalized signed dy in [-1, 1]
        
        # Time-to-intercept and predicted intercept position
        # Only calculate if ball is approaching
        ball_approaching = self.ball.dx > 0
        
        if ball_approaching and abs(self.ball.dx) > 0.1:
            # Calculate time to reach paddle's x position
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
        # AI is on right side, so mirror the state
        state[3] = -state[3]  # ball_dx_norm: flip sign (ball moving right -> moving left from perspective)
        state[7] = -state[7]  # signed_dx_norm: flip sign (ball to right -> to left from perspective)
        # ball_x_norm is already relative (0 = at paddle, 1 = away), so it's fine
        # All other features are symmetric (y positions, velocities, etc.)
        
        return state
    
    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        # Restart game if game over and space is pressed
        if self.game_over and keys[pygame.K_SPACE]:
            self.reset_game()
            return
        
        # Only allow paddle movement if game is not over
        if not self.game_over:
            # Left paddle controls (W/S) - Human player
            if keys[pygame.K_w]:
                self.paddle_left.move_up()
            if keys[pygame.K_s]:
                self.paddle_left.move_down()
            
            # AI controls right paddle
            state = self.get_state_for_ai()
            ai_action = self.ai.get_action_deterministic(state)  # Returns continuous action in [-1, 1]
            
            # Continuous action: -1 = up, 1 = down, 0 = stay
            if ai_action < -0.1:  # Move up (with small deadzone)
                move_amount = abs(ai_action) * self.paddle_right.speed
                self.paddle_right.rect.y = max(0, self.paddle_right.rect.y - move_amount)
            elif ai_action > 0.1:  # Move down (with small deadzone)
                move_amount = abs(ai_action) * self.paddle_right.speed
                self.paddle_right.rect.y = min(HEIGHT - PADDLE_HEIGHT, self.paddle_right.rect.y + move_amount)
            # Otherwise stay (action in [-0.1, 0.1] deadzone)
            
    def update(self):
        # Only update game if not over
        if self.game_over:
            return
            
        self.ball.move()
        
        # Check ball collision with paddles
        self.ball.check_collision(self.paddle_left)
        self.ball.check_collision(self.paddle_right)
        
        # Check if ball goes out of bounds
        if self.ball.rect.left <= 0:
            self.score_right += 1
            self.ball.reset()
        elif self.ball.rect.right >= WIDTH:
            self.score_left += 1
            self.ball.reset()
            
        # Check for winning condition
        if self.score_left >= WINNING_SCORE:
            self.game_over = True
            self.winner = "You"
        elif self.score_right >= WINNING_SCORE:
            self.game_over = True
            self.winner = "AI"
            
    def reset_game(self):
        """Reset the game to start a new match"""
        self.score_left = 0
        self.score_right = 0
        self.game_over = False
        self.winner = None
        self.ball.reset()
        self.paddle_left.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.paddle_right.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
            
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw center line
        pygame.draw.aaline(self.screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
        
        # Draw paddles
        self.paddle_left.draw(self.screen)
        self.paddle_right.draw(self.screen)
        
        # Draw ball
        self.ball.draw(self.screen)
        
        # Draw scores
        score_left_text = self.font.render(str(self.score_left), True, WHITE)
        score_right_text = self.font.render(str(self.score_right), True, WHITE)
        self.screen.blit(score_left_text, (WIDTH // 4, 20))
        self.screen.blit(score_right_text, (3 * WIDTH // 4, 20))
        
        # Draw labels
        player_label = self.small_font.render("YOU", True, WHITE)
        ai_label = self.small_font.render("AI", True, WHITE)
        self.screen.blit(player_label, (WIDTH // 4 - 30, 80))
        self.screen.blit(ai_label, (3 * WIDTH // 4 - 15, 80))
        
        # Draw game over message if game is over
        if self.game_over:
            game_over_text = self.font.render(f"{self.winner} Wins!", True, WHITE)
            restart_text = self.small_font.render("Press SPACE to restart", True, WHITE)
            text_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            self.screen.blit(game_over_text, text_rect)
            self.screen.blit(restart_text, restart_rect)
        else:
            # Draw controls hint
            controls_text = self.small_font.render("Controls: W (up) / S (down)", True, WHITE)
            self.screen.blit(controls_text, (WIDTH // 2 - 150, HEIGHT - 40))
        
        pygame.display.flip()
        
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Pong against trained AI')
    parser.add_argument('--model', type=str, default='best_model_final.pth',
                        help='Path to trained model (default: best_model_final.pth)')
    
    args = parser.parse_args()
    
    try:
        game = GameVsAI(args.model)
        game.run()
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train a model first or specify the correct path with --model")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

