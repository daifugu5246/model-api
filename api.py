from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vector_store_loader import select_and_combine_retrievers, load_all_vector_stores
from pydantic import BaseModel
from peft import PeftModel
import torch
import time
from fastapi import FastAPI

# device status
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:",device)
print("Device name:", torch.cuda.get_device_properties('cuda').name)
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

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

    retriever, docs_with_scores = select_and_combine_retrievers(
    symbol=None,
    quarter=None,
    report_vectorstore=report, 
    news_vectorstore=news, 
    instruction = retriever_prompt)
    return retriever


# API
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Finacial Articles Writing using Artificial Intelligence."}

@app.post("/generate")
def generate(req: PromptModel, symbol:str, quarter:str):
    
    start = time.time()
    
    # retrieve data
    retriever_prompt = format_retriever.format(req.instruction, req.data, req.event)
    retrieved = retriever(retriever_prompt,symbol,quarter)

    # generate
    prompt = format_prompt.format(req.instruction, req.data, req.event, retrieved)
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
    outputs = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    end = time.time()
    generated_time = end - start

    return {"message": response, "generated_time": generated_time}