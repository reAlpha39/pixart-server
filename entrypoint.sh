#!/bin/bash

# Pull the gemma2 model
echo "Pulling Gemma2 model..."
if ! huggingface-cli download bartowski/gemma-2-9b-it-GGUF --include "gemma-2-9b-it-Q4_K_M.gguf" --local-dir ./; then
    echo "Failed to pull gemma2 model."
    exit 1
fi

# Pull PixArt model from Hugging Face
echo "Pulling PixArt-Sigma-900M model..."
if ! huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir /app/PixArt-Sigma-900M; then
    echo "Failed to pull PixArt-Sigma-900M model."
    exit 1
fi

# Run the Python script
echo "Running the Python script..."
exec python3 main.py
