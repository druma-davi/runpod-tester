import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Definição do modelo
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# Variáveis globais para carregar o modelo apenas uma vez
model = None
tokenizer = None

def load_models():
    """Carrega o modelo e o tokenizador na memória (cold start)."""
    global model, tokenizer
    
    # Configuração para carregar o modelo em 4-bit e economizar VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Carregando modelo {MODEL_ID}...")
    
    # Carrega o tokenizador com permissão para executar código customizado
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Carrega o modelo com a configuração de 4-bit e permissão de código
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",  # Mapeia automaticamente para a GPU
        trust_remote_code=True  # ESSENCIAL para modelos da comunidade como este
    )
    print("Modelo carregado com sucesso.")

def handler(event):
    """Função principal que processa as requisições."""
    global model, tokenizer

    # Garante que o modelo está carregado (executa na primeira requisição)
    if model is None:
        load_models()

    # Extrai o texto do input JSON
    input_data = event["input"]
    user_text = input_data.get("text")

    if not user_text:
        return {"error": "Nenhum texto fornecido no campo 'text'."}

    # Transforma o texto simples em um formato de chat para o Qwen
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
    ]

    # Aplica o template de chat do modelo
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Prepara os inputs para o modelo
    model_inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device)

    # Gera a resposta
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024, # Aumente se precisar de respostas mais longas
        temperature=0.7,
        top_p=0.9
    )

    # Decodifica apenas a parte nova da resposta, removendo o prompt
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"response": response_text}

# Inicia o servidor do RunPod
runpod.serverless.start({"handler": handler})