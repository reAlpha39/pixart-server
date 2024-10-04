FROM ollama/ollama:latest

# Set the working directory
WORKDIR /root/.ollama

COPY entrypoint.sh /

ENTRYPOINT ["/usr/bin/bash", "/entrypoint.sh"]
