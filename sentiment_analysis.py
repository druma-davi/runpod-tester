import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

model = None
tokenizer = None

def load_models():
    global model, tokenizer
    
    # Configuração para rodar em 4-bit (pouca memória VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Carregando modelo {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Modelo carregado.")

def handler(event):
    global model, tokenizer

    if model is None:
        load_models()

    # 1. Pega o input exatamente como no seu exemplo JSON
    input_data = event["input"]
    user_text = input_data.get("text")

    if not user_text:
        return {"error": "Nenhum texto fornecido no campo 'text'."}

    # 2. Transforma o texto simples no formato de conversa do Qwen
    # O modelo funciona melhor se souber que é uma conversa User/System
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
    ]

    # 3. Formata o prompt (aplica <|im_start|>, etc)
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 4. Tokeniza e manda pra GPU
    model_inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device)

    # 5. Gera a resposta
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )

    # 6. Decodifica apenas a resposta nova (remove o prompt original)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Retorna no formato JSON simples
    return {"response": response_text}

runpod.serverless.start({"handler": handler})