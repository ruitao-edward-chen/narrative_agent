@echo off
echo Building frontend...
cd frontend
call npm install
call npm run build
cd ..

echo Installing Python dependencies...
pip install -r requirements.txt

echo Creating necessary directories...
if not exist agents mkdir agents

echo Starting Narrative Agent Marketplace...
python app.py 