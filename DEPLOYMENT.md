# Deployment Guide for Render

This guide will help you deploy the Pong AI game to Render.

## Prerequisites

1. A GitHub account
2. A Render account (sign up at https://render.com)
3. Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

Make sure your repository includes:
- `app.py` - Flask server
- `requirements.txt` - Python dependencies
- `templates/index.html` - Frontend
- `best_model_final.pth` - AI model file
- All game files (`pong_ai.py`, `ai_network.py`, etc.)

## Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Sign in or create an account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing this project

3. **Configure Service**
   - **Name**: `pong-ai` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Free (or choose paid for better performance)

4. **Environment Variables** (Optional)
   - No environment variables needed for basic setup
   - You can add `PYTHON_VERSION=3.9.0` if needed

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your application
   - Wait for deployment to complete (usually 2-5 minutes)

## Step 3: Access Your Game

Once deployed, Render will provide a URL like:
- `https://pong-ai.onrender.com`

Open this URL in your browser to play!

## Troubleshooting

### Model File Not Found
- Make sure `best_model_final.pth` is in your repository
- Check that the file is not in `.gitignore`

### Build Fails
- Check build logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### WebSocket Connection Issues
- Render free tier supports WebSockets
- Check that `flask-socketio` is properly installed
- Verify CORS settings in `app.py`

### Performance Issues
- Free tier may have cold starts (first request can be slow)
- Consider upgrading to paid tier for better performance
- Optimize game loop if needed

## Alternative: Using render.yaml

If you prefer configuration files, you can use the included `render.yaml`:

1. Push `render.yaml` to your repository
2. In Render dashboard, select "New +" → "Blueprint"
3. Connect your repository
4. Render will automatically detect and use `render.yaml`

## Notes

- Free tier services spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- Consider upgrading to paid tier for always-on service
- Model file size: Make sure your model file is under Render's limits

## Support

For issues:
1. Check Render logs in dashboard
2. Verify all files are in repository
3. Test locally first: `python app.py`
4. Check Render documentation: https://render.com/docs

