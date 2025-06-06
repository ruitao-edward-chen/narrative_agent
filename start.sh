#!/bin/bash

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Installing Python dependencies..."
pip install -r requirements.txt

mkdir -p agents

echo "Starting Narrative Agent Marketplace..."
python app.py 