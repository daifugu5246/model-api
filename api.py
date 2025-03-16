from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vector_store_loader import select_and_combine_retrievers, load_all_vector_stores
from pydantic import BaseModel
from peft import PeftModel
import torch
import time
import os

from fastapi import FastAPI, Query, Body, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


# device status
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:",device)
print("Device name:", torch.cuda.get_device_properties('cuda').name)
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
torch.cuda.empty_cache()

# load base model
model_name = "KBTG-Labs/THaLLE-0.1-7B-fa"
# model_name = "crumb/nano-mistral"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../model-cache")
base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="../model-cache", quantization_config=quantization_config)

# base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="../model-cache", torch_dtype=torch.bfloat16)
# base_model = base_model.to(device)

# load lora model
adapter_model = './lora_model_cot'
lora_model = PeftModel.from_pretrained(base_model, adapter_model)

# load model to device
lora_model = lora_model.to(device)

# load vector store
vector_store = load_all_vector_stores()

# format retriever
format_retriever = '''###Instruction: {}
###Input: 
Data:
{}
Event:
{}
'''

# format prompt
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
results = {}

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
def retriever(retriever_prompt,symbol,quarter):    
    report_vector_stores = vector_store[f"report_vector_stores"] #รับ dict จาก vectorstore_loader
    news_vector_stores = vector_store[f"news_vector_stores"]
    report = report_vector_stores[f"vector_store_{symbol}_{quarter}"]     #กำหนด vectorstore
    news = news_vector_stores[f"vector_store_{symbol}_news"]

    # print(report_vector_stores)

    retriever, docs_with_scores = select_and_combine_retrievers(
    symbol=None,
    quarter=None,
    report_vectorstore=report, 
    news_vectorstore=news, 
    instruction = retriever_prompt)
    return retriever

def generate_task(task_id: str, instruction: str, data: str, event: str, symbol: str, quarter: str):
    start = time.time()
    
    # Retrieve data
    retriever_prompt = format_retriever.format(instruction, data, event)
    retrieved = retriever(retriever_prompt, symbol, quarter)
    print("Retrieved Relevant Data.")

    # Generate output
    prompt = format_prompt.format(instruction, data, event, retrieved)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = lora_model.generate(
        inputs['input_ids'],
        max_new_tokens=2048,
        temperature=0.8,
        top_k=50,
        top_p=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode output
    outputs = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    end = time.time()
    generated_time = end - start

    print(f"Task {task_id} completed.")
    
    # Save result
    results[task_id] = {"message": response, "generated_time": generated_time}

# API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@app.get("/api")
def read_root():
    return {"message": "Finacial Articles Writing using Artificial Intelligence."}

@app.post("/api/generate")
async def generate(
    background_tasks: BackgroundTasks,
    req: PromptModel = Body(...),
    symbol: str = Query(...),
    quarter: str = Query(...)
):
    # Generate a unique task ID
    task_id = f"{symbol}_{quarter}_{int(time.time())}"

    # Start the task in the background
    background_tasks.add_task(generate_task, task_id, req.instruction, req.data, req.event, symbol, quarter)

    return {"task_id": task_id, "status": "Processing"}

@app.get("/api/result")
async def get_result(task_id: str):
    if task_id in results:
        return results.pop(task_id)  # Remove result after sending
    return {"status": "Processing"}

app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, timeout_keep_alive=10000, reload=True)
