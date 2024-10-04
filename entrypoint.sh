#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

ollama pull gemma2:latest

# Wait for Ollama process to finish.
wait $pid