import os
import sys

# CRITICAL: Set SDL video driver to dummy BEFORE importing pygame
# This prevents macOS from trying to create windows in background threads
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import torch
import numpy as np
from ai_network import PongNet
import threading
import time

# Initialize Pygame on main thread BEFORE any background threads
# This prevents macOS crashes when Pygame tries to initialize in background threads
import pygame
# Initialize Pygame on main thread with dummy driver
# This ensures it's fully initialized before any background threads try to use it
pygame.init()

from pong_ai import GameAI

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pong-ai-secret-key'

# Use threading mode instead of eventlet (more compatible, works on macOS and Render)
# async_mode can be 'threading', 'eventlet', 'gevent', or 'gevent_uwsgi'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load AI model
print("Loading AI model...")
try:
    checkpoint = torch.load('best_model_final.pth', weights_only=False, map_location='cpu')
    ai_model = PongNet(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size']
    )
    ai_model.load_state_dict(checkpoint['model_state_dict'])
    ai_model.eval()
    print(f"AI model loaded successfully (generation {checkpoint.get('generation', 'unknown')})")
except Exception as e:
    print(f"Error loading model: {e}")
    ai_model = None

# Game instances (one per client)
games = {}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Pong AI server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    if request.sid in games:
        del games[request.sid]

@socketio.on('start_game')
def handle_start_game():
    """Start a new game"""
    client_id = request.sid
    print(f"Starting game for client: {client_id}")
    
    # Create new game instance (headless) on main thread
    # This is critical: GameAI.__init__ must run on main thread to avoid macOS crashes
    try:
        game = GameAI(render=False, headless=True)
    except Exception as e:
        print(f"Error creating game: {e}")
        emit('error', {'message': f'Failed to start game: {str(e)}'})
        return
    
    games[client_id] = {
        'game': game,
        'running': True,
        'player_action': 0.0
    }
    
    # Get initial state
    state_left, state_right = game.reset()
    
    # Send initial game state
    emit('game_state', {
        'paddle_left_y': int(game.paddle_left.rect.centery),
        'paddle_right_y': int(game.paddle_right.rect.centery),
        'ball_x': int(game.ball.rect.centerx),
        'ball_y': int(game.ball.rect.centery),
        'ball_dx': float(game.ball.dx),
        'ball_dy': float(game.ball.dy),
        'score_left': game.score_left,
        'score_right': game.score_right,
        'game_over': game.game_over,
        'winner': game.winner
    })
    
    # Start game loop in background thread
    thread = threading.Thread(target=game_loop, args=(client_id,))
    thread.daemon = True
    thread.start()

@socketio.on('player_input')
def handle_player_input(data):
    """Handle player input (up/down)"""
    client_id = request.sid
    if client_id not in games:
        return
    
    action = data.get('action')  # 'up', 'down', or None
    
    if action == 'up':
        games[client_id]['player_action'] = -1.0  # Move up
    elif action == 'down':
        games[client_id]['player_action'] = 1.0  # Move down
    else:
        games[client_id]['player_action'] = 0.0  # Stay

def game_loop(client_id):
    """Game loop running in background thread - matches local pygame clock.tick(60) behavior"""
    if client_id not in games:
        return
    
    game_data = games[client_id]
    game = game_data['game']
    
    # Initialize player action
    game_data['player_action'] = 0.0
    
    target_fps = 60
    frame_time = 1.0 / target_fps  # ~0.016666 seconds per frame
    
    # Use time.perf_counter() for more precise timing
    last_time = time.perf_counter()
    
    try:
        while game_data['running'] and not game.game_over:
            current_time = time.perf_counter()
            elapsed = current_time - last_time
            
            # Only step once per frame (matching pygame clock.tick behavior)
            # Wait until enough time has passed for one frame
            if elapsed >= frame_time:
                # Get player action (left paddle)
                player_action = game_data.get('player_action', 0.0)
                
                # Get AI action (right paddle)
                if ai_model is not None:
                    state_right = game.get_state(is_left_paddle=False)
                    ai_action = ai_model.get_action_deterministic(state_right)
                else:
                    ai_action = 0.0
                
                # Step game exactly once (matching local behavior)
                game.step(player_action, ai_action)
                
                # Update last_time for next frame
                last_time = current_time
                
                # Send game state update
                socketio.emit('game_state', {
                    'paddle_left_y': int(game.paddle_left.rect.centery),
                    'paddle_right_y': int(game.paddle_right.rect.centery),
                    'ball_x': int(game.ball.rect.centerx),
                    'ball_y': int(game.ball.rect.centery),
                    'ball_dx': float(game.ball.dx),
                    'ball_dy': float(game.ball.dy),
                    'score_left': game.score_left,
                    'score_right': game.score_right,
                    'game_over': game.game_over,
                    'winner': game.winner
                }, room=client_id)
                
                # Check if game is over after step
                if game.game_over:
                    socketio.emit('game_over', {
                        'winner': game.winner,
                        'score_left': game.score_left,
                        'score_right': game.score_right
                    }, room=client_id)
                    break
            else:
                # Sleep until it's time for the next frame
                sleep_time = frame_time - elapsed
                if sleep_time > 0.001:  # Only sleep if more than 1ms
                    time.sleep(sleep_time)
            
            if game.game_over:
                break
    except Exception as e:
        print(f"Error in game loop for {client_id}: {e}")
        socketio.emit('error', {'message': str(e)}, room=client_id)

if __name__ == '__main__':
    import socket
    
    # Get port from environment or use default
    base_port = int(os.environ.get('PORT', 5000))
    
    # Find an available port
    def find_free_port(start_port):
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None
    
    port = find_free_port(base_port)
    
    if port is None:
        print(f"Error: Could not find an available port starting from {base_port}")
        sys.exit(1)
    
    if port != base_port:
        print(f"Port {base_port} is in use. Using port {port} instead.")
    
    print(f"Starting server on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    
    # Run with threading mode (no eventlet needed)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)

