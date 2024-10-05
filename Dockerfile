FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install CUDA dependencies and other necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirement.txt /app/
RUN pip3 install -r requirement.txt

# Ensure the generated_images directory exists
RUN mkdir -p /app/generated_images

# Copy only main.py and the modules folder, exclude models as they are mounted from the host
COPY main.py /app/
COPY modules /app/modules/

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the API port
EXPOSE 8000

# Use entrypoint.sh to run the sequence of commands
ENTRYPOINT ["/app/entrypoint.sh"]
