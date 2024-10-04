FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install CUDA dependencies and other necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface-cli
RUN pip3 install huggingface_hub

# Copy requirements.txt and install Python dependencies
COPY requirement.txt /app/
RUN pip3 install -r requirement.txt

# Download the Hugging Face models
# RUN huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir /app/PixArt-Sigma-900M

# Ensure the generated_images directory exists
RUN mkdir -p /app/generated_images

# Copy only main.py and the modules folder, exclude models as they are mounted from the host
COPY main.py /app/
COPY modules /app/modules/

# Expose the API port
EXPOSE 8000

# Command to run the API
CMD ["python3", "main.py"]
