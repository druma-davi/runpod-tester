# Use a imagem oficial do RunPod com Python 3.10 e CUDA
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Define a pasta de cache para garantir que o modelo pré-baixado seja encontrado
ENV HF_HOME="/app/model_cache"

# Copia e instala as dependências
COPY requirements.txt .

# --- LINHA CORRIGIDA ---
# Força a atualização das bibliotecas para a versão mais recente
RUN pip install --upgrade pip && \
    pip install --upgrade --no-cache-dir transformers accelerate bitsandbytes && \
    pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD DO MODELO ---
# (Esta parte já estava correta)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_id = 'Qwen/Qwen3-4B-Thinking-2507'; \
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)"

# Copia o código da aplicação
COPY sentiment_analysis.py .

# Comando para iniciar a aplicação
CMD ["python", "-u", "sentiment_analysis.py"]