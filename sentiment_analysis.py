import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# O modelo solicitado: Qwen 2.5 3B (Versão Instruct é melhor para chat)
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

model = None
tokenizer = None

def load_models():
    """Carrega o modelo na memória GPU com otimização 4-bit."""
    global model, tokenizer
    
    print(f"--- Carregando {MODEL_ID} em 4-bit... ---")
    
    # Configuração para usar pouquíssima memória VRAM (roda rápido e barato)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Carrega Tokenizador
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # Carrega Modelo
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto", # Usa a GPU automaticamente
            trust_remote_code=True
        )
        print("--- Modelo carregado com sucesso! ---")
    except Exception as e:
        print(f"Erro fatal ao carregar modelo: {e}")
        raise e

def handler(event):
    global model, tokenizer

    # Cold Start: Se o modelo não estiver na RAM, carrega agora
    if model is None:
        load_models()

    # 1. Ler o input do JSON (exatamente como no seu exemplo "text")
    input_data = event.get("input", {})
    user_text = input_data.get("text")

    if not user_text:
        return {"error": "Por favor forneça 'text' dentro do objeto 'input'."}

    # 2. Criar estrutura de Chat (User/System)
    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": user_text}
    ]

    # 3. Formatar o prompt para o modelo entender que é um chat
    text_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 4. Processar na GPU
    model_inputs = tokenizer([text_formatted], return_tensors="pt").to(model.device)

    # 5. Gerar a resposta
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512, # Tamanho máximo da resposta
        temperature=0.7,
        top_p=0.9
    )

    # 6. Limpar a resposta (remover o input original e deixar só a fala do bot)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Retorna no formato JSON
    return {"response": response}

# Inicia o servidor
runpod.serverless.start({"handler": handler})