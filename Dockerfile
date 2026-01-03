# Usa a imagem oficial do RunPod que já tem CUDA e Python 3.10
# (Muito melhor que python:3.9-slim para IA)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho
WORKDIR /app

# --- CRÍTICO: Define o cache na pasta app para não estourar o disco do sistema ---
ENV HF_HOME="/app/model_cache"

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD DO MODELO ---
# Baixa o Qwen2.5-3B-Instruct agora para não baixar na hora que ligar
RUN python -c "import torch; \
    from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_id = 'Qwen/Qwen3-4B-Thinking-2507'; \
    print(f'Baixando {model_id}...'); \
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)"

# Copia o seu código (mantive o nome sentiment_analysis.py como no seu original)
COPY sentiment_analysis.py .

# Comando de execução com -u para ver logs em tempo real
CMD ["python", "-u", "sentiment_analysis.py"]