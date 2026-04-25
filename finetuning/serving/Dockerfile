# Build context: finetuning/
# Usage: docker build -f serving/Dockerfile -t cld-judge .
FROM vllm/vllm-openai:latest

RUN mkdir -p /adapters

ENV CLD_BASE_MODEL=""
ENV CLD_PRESET=""
ENV CLD_LORA_PATH=""
ENV CLD_LORA_NAME="cld_judge"
ENV CLD_HOST="0.0.0.0"
ENV CLD_PORT="8000"
ENV CLD_TENSOR_PARALLEL_SIZE=""
ENV CLD_MAX_MODEL_LEN="2048"
ENV CLD_QUANTIZATION=""
ENV CLD_DTYPE="auto"
ENV CLD_GPU_MEM_UTIL="0.90"
ENV HF_HOME="/root/.cache/huggingface"

COPY src/serving/serve_judge_vllm.py /app/serve_judge_vllm.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:${CLD_PORT}/health || exit 1

ENTRYPOINT ["python", "/app/serve_judge_vllm.py"]
