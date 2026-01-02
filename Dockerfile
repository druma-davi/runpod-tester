# Use a imagem oficial do RunPod com Python 3.10 e CUDA (essencial para GPU)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho
WORKDIR /app

# Define a pasta de cache DENTRO da imagem para evitar erros de espaço em disco
# e garantir que o modelo pré-baixado seja encontrado.
ENV HF_HOME="/app/model_cache"

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD DO MODELO (COM O FIX 'trust_remote_code') ---
# Baixa o modelo durante o build para que o servidor inicie instantaneamente.
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_id = 'Qwen/Qwen3-4B-Thinking-2507'; \
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)"

# Copia o código da aplicação para o container
COPY sentiment_analysis.py .

# Comando para iniciar a aplicação quando o container rodar
# O "-u" é para os logs aparecerem em tempo real no RunPod
CMD ["python", "-u", "sentiment_analysis.py"]