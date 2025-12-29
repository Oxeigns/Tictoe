# TicToe Telegram Bot

Premium-quality async TicTacToe bot designed for Heroku worker dynos.

## Requirements
- Python 3.11.x
- Environment variables: `BOT_TOKEN`, `MONGO_URL`, `OWNER_ID`, `API_ID`, `API_HASH`.
- Optional overrides: `JOIN_TIMEOUT_SEC`, `TURN_TIMEOUT_SEC`, `CALLBACK_THROTTLE_SEC`.

## Running locally
1. Create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Export the required environment variables or create a `.env` file for local development.
3. Start the bot:
   ```bash
   python app.py
   ```

## Deploying to Heroku
1. Ensure the Heroku CLI is installed and you are logged in.
2. Create the app and set config vars:
   ```bash
   heroku create your-app-name
   heroku config:set BOT_TOKEN=... MONGO_URL=... OWNER_ID=... API_ID=... API_HASH=...
   ```
3. Push the repository:
   ```bash
   git push heroku main
   ```
4. Scale the worker process:
   ```bash
   heroku ps:scale worker=1
   ```

## Common troubleshooting
- Verify MongoDB URI allows connections from Heroku.
- Ensure the bot is invited to target groups and has permission to send/edit messages.
- Use `/setloggroup <chat_id>` as an admin to receive structured event logs.
