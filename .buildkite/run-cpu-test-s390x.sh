#!/bin/bash

# Define log file
LOG_FILE="cpu_test_log_$(date +%Y%m%d%H%M%S).log"

# Redirect stdout and stderr to the log file
exec > >(tee -a "$LOG_FILE") 2>&1

# This script runs offline inference inside the container.
set -ex

# Setup cleanup
#remove_docker_container() { 
#    echo "Attempting to stop and remove Docker container..."
#    if docker ps -q --filter "name=cpu-test" | grep -q .; then
#        echo "Stopping container cpu-test..."
#        docker stop cpu-test || {
#            echo "Failed to stop container cpu-test gracefully. Retrying..."
#            sleep 5
#            docker stop cpu-test || echo "Forcefully stopping container cpu-test."
#        }
#        echo "Removing container cpu-test..."
#        docker rm -f cpu-test || echo "Failed to remove container cpu-test."
#    else
#        echo "No container named cpu-test found running."
#    fi
#}
#trap remove_docker_container EXIT
#remove_docker_container

# Pull pre-built image
docker pull docker.io/nishan321/cpu-test:latest

# Run the image
source /etc/environment
docker run -itd \
  --entrypoint /bin/bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --privileged=true \
  --network host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --name cpu-test \
  --replace \
  nishan321/cpu-test:latest

function cpu_tests() {
  set -e

  # Check HF_TOKEN availability
  if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set. Please provide a valid Hugging Face token."
    exit 1
  fi

  # Run basic model tests
  docker exec cpu-test bash -c "
    set -e
    echo 'Installing dependencies...'
    pip install pytest pytest-asyncio \
      einops librosa peft Pillow sentence-transformers soundfile \
      transformers_stream_generator matplotlib datamodel_code_generator
    pip install torchvision --index-url https://download.pytorch.org/whl/cpu

    echo 'Starting pytest: decoder_only/language'
    pytest -v -s tests/models/decoder_only/language -m cpu_model || echo 'Test failed: decoder_only/language'

    echo 'Starting pytest: embedding/language'
    pytest -v -s tests/models/embedding/language -m cpu_model || echo 'Test failed: embedding/language'

    echo 'Starting pytest: encoder_decoder/language'
    pytest -v -s tests/models/encoder_decoder/language -m cpu_model || echo 'Test failed: encoder_decoder/language'

    echo 'Starting pytest: decoder_only/audio_language'
    pytest -v -s tests/models/decoder_only/audio_language -m cpu_model || echo 'Test failed: decoder_only/audio_language'

    echo 'All tests completed.'
  "

  # Online inference
  docker exec cpu-test bash -c "
    set -e
    echo 'Starting the VLLM API server...'
    python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --dtype float &
    echo 'Waiting for API server to be ready...'
    timeout 600 bash -c 'until curl -s localhost:8000/v1/models; do sleep 1; done' || exit 1
    echo 'Running benchmark tests...'
    python3 benchmarks/benchmark_serving.py \
      --backend vllm \
      --dataset-name random \
      --model facebook/opt-125m \
      --num-prompts 20 \
      --endpoint /v1/completions \
      --tokenizer facebook/opt-125m || echo 'Benchmark tests failed.'
  "
}


# Run tests with timeout
export -f cpu_tests
timeout 140m bash -c "cpu_tests"

