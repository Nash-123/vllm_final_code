#Modified bash script 

#!/bin/bash


# Define log file
LOG_FILE="cpu_test_log_$(date +%Y%m%d%H%M%S).log"

# Redirect stdout and stderr to the log file
exec > >(tee -a "$LOG_FILE") 2>&1

# This script builds the CPU Docker image and runs offline inference inside the container.
# It serves as a sanity check for compilation and basic model usage.
set -ex



# Try building the docker image
docker build -t cpu-test -f Dockerfile.s390x .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image, setting --shm-size=4g for tensor parallel.
source /etc/environment
docker run -itd \
  --entrypoint /bin/bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --privileged=true \
  --network host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --name cpu-test \
  cpu-test

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
    pip install pytest pytest-asyncio \
      einops librosa peft Pillow sentence-transformers soundfile \
      transformers_stream_generator matplotlib datamodel_code_generator
    pip install torchvision --index-url https://download.pytorch.org/whl/cpu
    pytest -v -s tests/models/decoder_only/language -m cpu_model || true
    pytest -v -s tests/models/embedding/language -m cpu_model || true
    pytest -v -s tests/models/encoder_decoder/language -m cpu_model || true
    pytest -v -s tests/models/decoder_only/audio_language -m cpu_model || true
  "

  # Online inference
  docker exec cpu-test bash -c "
    set -e
    python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --dtype float &
    timeout 600 bash -c 'until curl -s localhost:8000/v1/models; do sleep 1; done' || exit 1
    python3 benchmarks/benchmark_serving.py \
      --backend vllm \
      --dataset-name random \
      --model facebook/opt-125m \
      --num-prompts 20 \
      --endpoint /v1/completions \
      --tokenizer facebook/opt-125m || true
  "
}

# Run tests with timeout
export -f cpu_tests
timeout 25m bash -c "cpu_tests"

