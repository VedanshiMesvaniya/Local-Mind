Step 3: Create ollama-entrypoint.sh
Ollama doesn't download models automatically on startup. This script forces it to pull your models the first time the container runs.
Create a file named ollama-entrypoint.sh:
bash

#!/bin/bash
# Start the Ollama service in the background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done

echo "Ollama is ready. Pulling required models..."

# Pull the models needed for your pipeline
ollama pull nomic-embed-text
ollama pull phi4-mini:latest 

echo "Models downloaded successfully!"

# Keep the container running
wait


### chmod +x ollama-entrypoint.sh