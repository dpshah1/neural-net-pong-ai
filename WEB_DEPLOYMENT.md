# Web Deployment Quick Start

## Local Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   ```
   http://localhost:5000
   ```

4. **Play:**
   - Click "Start Game"
   - Use `W` and `S` keys to move your paddle
   - Play against the AI!

## Deploy to Render

### Option 1: Using Render Dashboard

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add web deployment"
   git push origin main
   ```

2. **Go to Render:**
   - Visit https://dashboard.render.com
   - Sign up/login (free account)

3. **Create Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure:**
   - **Name**: `pong-ai` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Free

5. **Deploy:**
   - Click "Create Web Service"
   - Wait 2-5 minutes for deployment
   - Your game will be live at: `https://pong-ai.onrender.com`

### Option 2: Using render.yaml

1. **Push code with render.yaml:**
   ```bash
   git add render.yaml
   git commit -m "Add Render config"
   git push
   ```

2. **In Render Dashboard:**
   - Click "New +" → "Blueprint"
   - Connect repository
   - Render will auto-detect `render.yaml`

## Important Notes

- **Model File**: Make sure `best_model_final.pth` is in your repository (not in .gitignore)
- **Free Tier**: Services spin down after 15 min inactivity (first request may be slow)
- **WebSocket**: Render free tier supports WebSockets
- **Port**: App automatically uses `PORT` environment variable (Render sets this)

## Troubleshooting

**Build fails:**
- Check that `best_model_final.pth` exists in repo
- Verify all dependencies in `requirements.txt`
- Check Render build logs

**WebSocket not working:**
- Ensure `eventlet` is installed (in requirements.txt)
- Check Render logs for errors

**Game not loading:**
- Check browser console for errors
- Verify server is running (check Render logs)
- Test locally first: `python app.py`

## File Structure

```
pong-game-nn/
├── app.py                 # Flask server
├── templates/
│   └── index.html        # Frontend
├── requirements.txt       # Dependencies
├── render.yaml           # Render config
├── Procfile              # Process file
├── best_model_final.pth  # AI model (required!)
└── ... (game files)
```

## Next Steps

After deployment:
1. Share your Render URL
2. Play against your AI!
3. Update model by retraining and redeploying

