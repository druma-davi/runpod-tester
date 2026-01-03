import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

model = None
tokenizer = None

def load_models():
    global model, tokenizer
    
    # Quantização 4-bit para o Qwen3
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Qwen3 prefere bfloat16
        bnb_4bit_use_double_quant=True,
    )

    print(f"--- Inicializando Qwen 3 Architecture: {MODEL_ID} ---")
    
    # trust_remote_code=True é obrigatório para Qwen3
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" # Ativa a performance do Qwen3
    )
    print("--- Qwen 3 carregado com sucesso ---")

def handler(event):
    global model, tokenizer

    if model is None:
        load_models()

    input_data = event.get("input", {})
    user_text = input_data.get("text")

    if not user_text:
        return {"error": "No 'text' provided in input."}

    # Formatando para a estrutura de Chat do Qwen 3
    messages = [
        {"role": "system", "content": "You are a Qwen 3 Thinking model. Explain your reasoning process clearly."},
        {"role": "user", "content": user_text}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Geração otimizada para modelos Thinking (geralmente geram mais tokens)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.6, # Modelos Thinking funcionam melhor com temp ligeiramente baixa
        top_p=0.95,
        do_sample=True
    )

    # Remove o prompt dos tokens gerados
    generated_tokens = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"response": response}

runpod.serverless.start({"handler": handler})