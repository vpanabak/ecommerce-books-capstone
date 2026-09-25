# Build Docker image
docker build -t zepto-support-assistant .

# Run Docker container locally
docker run -p 7860:7860 zepto-support-assistant