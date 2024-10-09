#!/bin/bash

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

# Pull the gemma2:2b model
echo "Pulling the gemma2:2b model..."
if ! ollama pull gemma2:2b; then
    echo "Failed to pull gemma2:2b model."
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

# Check if DOWNLOAD_PIXART_XL is set to 1
if [ "$DOWNLOAD_PIXART_XL" = "1" ]; then
    # Pull PixArt-Sigma-XL-2-1024-MS model from Hugging Face
    echo "Pulling PixArt-Sigma-XL-2-1024-MS model..."
    if [ ! -d "/app/PixArt-Sigma-XL-2-1024-MS" ]; then
        if ! huggingface-cli download PixArt-alpha/PixArt-Sigma-XL-2-1024-MS --local-dir /app/PixArt-Sigma-XL-2-1024-MS; then
            echo "Failed to pull PixArt-Sigma-XL-2-1024-MS model."
            exit 1
        fi
    else
        echo "PixArt-Sigma-XL-2-1024-MS model is already downloaded."
    fi

    # Pull pixart_sigma_sdxlvae_T5_diffusers model from Hugging Face
    echo "Pulling pixart_sigma_sdxlvae_T5_diffusers model..."
    if [ ! -d "/app/pixart_sigma_sdxlvae_T5_diffusers" ]; then
        if ! huggingface-cli download PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir /app/pixart_sigma_sdxlvae_T5_diffusers; then
            echo "Failed to pull pixart_sigma_sdxlvae_T5_diffusers model."
            exit 1
        fi
    else
        echo "pixart_sigma_sdxlvae_T5_diffusers model is already downloaded."
    fi
else
    echo "Skipping PixArt-Sigma-XL-2-1024-MS and pixart_sigma_sdxlvae_T5_diffusers models download."
fi

# Run the Python script
echo "Running the Python script..."
exec python3 main.py