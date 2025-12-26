import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Constants
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

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong - Two Players")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)
        
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
        
    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        # Restart game if game over and space is pressed
        if self.game_over and keys[pygame.K_SPACE]:
            self.reset_game()
            return
        
        # Only allow paddle movement if game is not over
        if not self.game_over:
            # Left paddle controls (W/S)
            if keys[pygame.K_w]:
                self.paddle_left.move_up()
            if keys[pygame.K_s]:
                self.paddle_left.move_down()
                
            # Right paddle controls (Up/Down arrows)
            if keys[pygame.K_UP]:
                self.paddle_right.move_up()
            if keys[pygame.K_DOWN]:
                self.paddle_right.move_down()
            
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
            self.winner = "Left Player"
        elif self.score_right >= WINNING_SCORE:
            self.game_over = True
            self.winner = "Right Player"
            
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
        
        # Draw game over message if game is over
        if self.game_over:
            game_over_text = self.font.render(f"{self.winner} Wins!", True, WHITE)
            restart_text = self.small_font.render("Press SPACE to restart", True, WHITE)
            text_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            self.screen.blit(game_over_text, text_rect)
            self.screen.blit(restart_text, restart_rect)
        else:
            # Draw controls hint only when game is active
            controls_text = self.small_font.render("Left: W/S  |  Right: ↑/↓", True, WHITE)
            self.screen.blit(controls_text, (WIDTH // 2 - 120, HEIGHT - 40))
        
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
    game = Game()
    game.run()

