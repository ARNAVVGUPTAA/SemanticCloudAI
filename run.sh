#!/bin/bash
set -e

echo "====================================="
echo " Starting SemanticCloudAI (Local)    "
echo "====================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Please install it to continue."
    exit 1
fi



# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies if not installed
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Export environment variables for Local Mode
export LOCAL_MODE=1
export CHROMA_DATA_PATH="./chroma_data"
# Setting LOCAL_MODE=1 triggers eager Celery tasks and SQLite so Redis/Postgres are not needed.

# Start Uvicorn
echo "Database initialized. Starting FastAPI Server..."
echo "API will be available at http://localhost:8000"
echo "Press Ctrl+C to stop."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
