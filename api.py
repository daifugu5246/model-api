from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import time
from fastapi import FastAPI

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device name:", torch.cuda.get_device_properties('cuda').name)
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
print(f'torch version: {torch.version}')

# load base model
model_name = "KBTG-Labs/THaLLE-0.1-7B-fa"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../model-cache")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="../model-cache", quantization_config=quantization_config)

# load lora model
adapter_model = './lora_model_cot'
lora_model = PeftModel.from_pretrained(base_model, adapter_model)

# load model to device
lora_model = lora_model.to(device)

format_prompt = '''###Instruction: {}
###Input: 
Data:
{}
Event:
{}
###Retrieved:
{}
###Output:

'''

prompt = 'PTT คือ บริษัทเกี่ยวกับอะไร?'

inputs = tokenizer(prompt, return_tensors='pt').to(device)

outputs = lora_model.generate(
    inputs['input_ids'],
    max_new_tokens=2048,
    temperature = 0.8,
    top_k = 50,
    top_p = 0.8,
    repetition_penalty = 1.1,
    pad_token_id=tokenizer.eos_token_id
)

outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs) ]
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)

# API
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Finacial Articles Writing using Artificial Intelligence."}

@app.get("/generate")
def generate(instruction: str, data: str, event: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = lora_model.generate(
        inputs['input_ids'], 
        max_new_tokens=2048,  # คุณสามารถเปลี่ยนค่านี้เพื่อควบคุมความยาวของผลลัพธ์
        temperature=0.8,     # ควบคุมความคิดสร้างสรรค์/ความสุ่ม (ค่าน้อย = สุ่มน้อย, ค่ามาก = สุ่มมาก)
        top_k=50,            # การสุ่มจาก top_k โทเคน
        top_p=0.8,           # การสุ่มแบบ nucleus sampling
        repetition_penalty=1.1,

    )
    outputs = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"message": response}