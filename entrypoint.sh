#!/bin/bash

# Install Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    if ! curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.3.10 sh; then
        echo "Failed to install Ollama."
        exit 1
    fi
else
    echo "Ollama is already installed."
fi

# Start Ollama service in the background
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait a moment to ensure Ollama is ready
echo "Waiting for Ollama to start..."
sleep 5

# Check if Ollama is running
if ! kill -0 $OLLAMA_PID > /dev/null 2>&1; then
    echo "Ollama service failed to start."
    exit 1
fi

# Pull the gemma2 model
echo "Pulling the gemma2 model..."
if ! ollama pull gemma2; then
    echo "Failed to pull gemma2 model."
    exit 1
fi

# Pull PixArt model from Hugging Face
echo "Pulling PixArt-Sigma-900M model..."
if [ ! -d "/app/PixArt-Sigma-900M" ]; then
    if ! huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir /app/PixArt-Sigma-900M; then
        echo "Failed to pull PixArt-Sigma-900M model."
        exit 1
    fi
else
    echo "PixArt-Sigma-900M model is already downloaded."
fi

# Run the Python script
echo "Running the Python script..."
exec python3 main.py