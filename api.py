from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel
from peft import PeftModel
import time
from fastapi import FastAPI

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load base model
model_name = "KBTG-Labs/THaLLE-0.1-7B-fa"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../../model-cache")
base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="../../model-cache", torch_dtype=torch.bfloat16)

# load lora model
adapter_model = './lora_model_cot'
lora_model = PeftModel.from_pretrained(base_model, adapter_model)

# load model to device
lora_model = lora_model.to(device)

# load vectorstore
vector_store = load_all_vector_stores()

# mockup data
symbol = "PTT"
quarter ="Q1_66"


format_prompt = '''###Instruction: {}
###Input: 
Data:
{}
Event:
{}
###Retrived:
{}
###Output:

'''

""" prompt = 'PTT คือ บริษัทเกี่ยวกับอะไร?'

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
print(response) """

# prompt
class PromptModel(BaseModel):
    instruction: str
    data: str
    event: str

# ฝากแทนเขียน function นี้ด้วยนะครับ
def retriever(retriever_prompt):
    pass


# API
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Finacial Articles Writing using Artificial Intelligence."}

@app.post("/generate")
def generate(req: PromptModel):
    
    start = time.time()
    
    # retrieve data
    retriever_prompt = format_retriever.format(req.instruction, req.data, req.event)
    retrieved = retriever(retriever_prompt)

    # generate
    prompt = format_prompt.format(req.instruction, req.data, req.event, retrieved)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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