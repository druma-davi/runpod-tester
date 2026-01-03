# Imagem base com CUDA 12+ para suportar as otimizações do Qwen 3
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Instala dependências do sistema necessárias para compilar extensões do Qwen 3
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Cache do HuggingFace em local persistente
ENV HF_HOME="/app/model_cache"
ENV TRANSFORMERS_CACHE="/app/model_cache"

COPY requirements.txt .

# Instalação forçada das versões de 2026 para arquitetura Thinking
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Instala Flash Attention 2 (O Qwen 3 EXIGE isso para a arquitetura de atenção)
RUN pip install flash-attn --no-build-isolation

# PRE-DOWNLOAD: Agora com ambiente preparado (git + dependências instaladas)
RUN python -c "import torch; \
    from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_id = 'Qwen/Qwen3-4B-Thinking-2507'; \
    print(f'Carregando {model_id}...'); \
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cpu')"

COPY sentiment_analysis.py .

CMD ["python", "-u", "sentiment_analysis.py"]