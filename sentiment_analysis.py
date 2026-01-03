import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

model = None
tokenizer = None

def load_models():
    global model, tokenizer
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Carregando modelo {MODEL_ID}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("Modelo carregado com sucesso.")

def handler(event):
    global model, tokenizer

    if model is None:
        load_models()

    input_data = event["input"]
    user_text = input_data.get("text")

    if not user_text:
        return {"error": "Nenhum texto fornecido no campo 'text'."}

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
    ]

    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9
    )

    # --- CORREÇÃO DE BUG AQUI ---
    # Era model_inputs.ids, o correto é model_inputs.input_ids
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"response": response_text}

runpod.serverless.start({"handler": handler})