#!/bin/bash

# Start Ollama serve in the background
ollama serve &

# Wait a moment to ensure Ollama is ready
sleep 5

# Pull the gemma2 model
ollama pull gemma2

# pull pixart model
huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir /app/PixArt-Sigma-900M

# Run the Python script
python3 main.py
