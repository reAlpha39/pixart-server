FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install CUDA dependencies and other necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev libpython3-dev python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    build-essential git curl pciutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install llama-cpp-python
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python

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
