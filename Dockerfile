FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install CUDA dependencies and other necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirement.txt .
RUN pip3 install -r requirements.txt

# Ensure the generated_images directory exists
RUN mkdir -p generated_images

# Copy only main.py and the modules folder.
COPY main.py .
COPY modules/ modules/

# Copy the entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose the API port
EXPOSE 8000

# Use entrypoint.sh to run the sequence of commands
ENTRYPOINT ["./entrypoint.sh"]
