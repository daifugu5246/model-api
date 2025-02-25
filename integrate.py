from transformers import AutoModelForCausalLM, AutoTokenizer
from vector_store_loader import select_and_combine_retrievers, load_all_vector_stores
from peft import PeftModel
from fastapi import FastAPI
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# load base model
model_name = "KBTG-Labs/THaLLE-0.1-7B-fa"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../model-cache")
base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="../model-cache", torch_dtype=torch.bfloat16)

# load lora model
adapter_model = './lora_model_cot'
lora_model = PeftModel.from_pretrained(base_model, adapter_model)

# load model to device
lora_model = lora_model.to(device)

# load vector
vector_store = load_all_vector_stores()




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

symbol = "PTT"
quarter = "Q2_67"

#รับ dict จาก vectorstore_loader
report_vector_stores = vector_store[f"report_vector_stores"]
news_vector_stores = vector_store[f"news_vector_stores"]

#กำหนด vectorstore
report = report_vector_stores[f"vector_store_{symbol}_{quarter}"]
news = news_vector_stores[f"vector_store_{symbol}_news"]

#Part Test
#############################################################################################################################################################################################################################################################################################
# symbol = "PTT"
# quarter = "Q2_67"
# instruction = "เขียนบทวิเคราะห์หุ้นรายตัวของ GULF โดยมีประโยคสั้นชี้แจงเกี่ยวกับมุมมองต่อแผนควบรวมกิจการ โดยให้เนื้อหาแต่ละย่อหน้ากล่าวถึง มุมมองต่อช้อเสนอการควบรวมของ GULF-INTUCH, ประเด็นสำคัญจากการประชุมเสมือนจริง, การประเมินมูลค่าหุ้น, ความเสี่ยง, ให้คำแนะนำการลงทุนและราคาเป้าหมายหุ้น โดยใช้ข้อมูลที่มอบให้ต่อไปนี้ในการเขียนบทวิเคราะห์"
# data = """- ราคาเป้าหมายของ GULF: 57.0 บาท (เพิ่มขึ้นจาก 56.5 บาท)
# - คาดการณ์กำไรหลักปี 2567: 1.76 หมื่นล้านบาท (เติบโต 13% YoY)
# - อัพไซด์จาก PDP ใหม่: 7% ต่อประมาณการมูลค่ายุติธรรม
# - กำลังการผลิตที่คาดว่า GULF จะชนะประมูล: 8.8 GW (30% ของกำลังการผลิตใหม่)
# - สัดส่วนกำลังการผลิต SPP ของ GULF: 16% ของกำลังการผลิตรวม
# - ราคาหุ้น GULF เพิ่มขึ้น 2.3% ในช่วง 4 เดือนที่ผ่านมา (น้อยกว่า GPSC 42.3% และ BGRIM 26.1%)"""
# event = """- GULF ประกาศแผนการปรับโครงสร้างผู้ถือหุ้นจากการควบรวมกิจการ GULF-INTUCH เมื่อวันที่ 16 ก.ค.
# - GULF จัดการประชุมเสมือนจริงเมื่อวันที่ 17 ก.ค. เพื่ออธิบายแผนการควบรวม
# - การควบรวมคาดว่าจะนำไปสู่:
#   1. การลดอัตราส่วนหนี้สินสุทธิต่อทุน
#   2. การเพิ่มรายได้ประมาณ 2 พันล้านบาทต่อปี
# - ยังไม่มีการตัดสินใจเกี่ยวกับการถือหุ้น 9% ของ Singtel ใน NewCo
# - การลดอัตราส่วนหนี้สินสุทธิอาจส่งผลให้หน่วยงานจัดอันดับเครดิตพิจารณาปรับเพิ่มอันดับเครดิตของ GULF
# - คาดการณ์ว่า GULF จะได้รับประโยชน์จากการประมูลในอนาคตภายใต้แผน PDP ใหม่ """


# #รับ dict จาก vectorstore_loader
# report_vector_stores = vector_store[f"report_vector_stores"]
# news_vector_stores = vector_store[f"news_vector_stores"]

# #กำหนด vectorstore
# report = report_vector_stores[f"vector_store_{symbol}_{quarter}"]
# news = news_vector_stores[f"vector_store_{symbol}_news"]

# #select and combined data from vectorstore
# retriever, docs_with_scores = select_and_combine_retrievers(
# symbol=None,
# quarter=None,
# report_vectorstore=report, 
# news_vectorstore=news, 
# instruction = instruction  )
# print("finished Retriever")

# prompt = format_prompt.format(instruction, data, event, retriever)

# inputs = tokenizer(prompt, return_tensors='pt').to(device)
# print("start generate output")
# outputs = lora_model.generate(
#     inputs['input_ids'],
#     max_new_tokens=2048,
#     temperature = 0.8,
#     top_k = 50,
#     top_p = 0.8,
#     repetition_penalty = 1.1,
#     pad_token_id=tokenizer.eos_token_id
# )

# outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs) ]
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# print(response)
#############################################################################################################################################################################################################################################################################################

# API
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Finacial Articles Writing using Artificial Intelligence."}

@app.get("/generate")
def generate(instruction: str, data: str, event: str):

    prompt = format_prompt.format(instruction, data, event, retriever)

    #select and combined data from vectorstore
    retriever, docs_with_scores = select_and_combine_retrievers(
    symbol=None,
    quarter=None,
    report_vectorstore=report, 
    news_vectorstore=news, 
    instruction = instruction  )
    print("finished Retriever")

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


    
