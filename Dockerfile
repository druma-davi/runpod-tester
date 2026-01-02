# 1. Usar a imagem oficial do RunPod (já vem com Python 3.10 e CUDA 12.1 instalados)
# Isso é MUITO mais rápido que usar python:3.9-slim e tentar instalar drivers manualmente.
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho
WORKDIR /app

# 2. Copia o requirements primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Atualiza o pip e instala as dependências
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. PRE-DOWNLOAD DO MODELO (Sua solicitação)
# Isso baixa o Qwen2.5-3B para dentro da imagem durante o build.
# Assim, o container inicia instantaneamente sem baixar nada.
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_id = 'Qwen/Qwen3-4B-Thinking-2507'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModelForCausalLM.from_pretrained(model_id)"

# 4. Copia o código da aplicação (seu script python atualizado)
# Certifique-se de que o nome do arquivo aqui corresponda ao seu arquivo (ex: handler.py)
COPY sentiment_analysis.py .

# 5. Comando de execução
# O flag "-u" é importante para ver os logs em tempo real no RunPod
CMD ["python", "-u", "sentiment_analysis.py"]