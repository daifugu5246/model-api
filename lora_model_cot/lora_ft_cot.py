from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import Dataset, concatenate_datasets
import torch
import torch.nn as nn
import math
import random

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())
torch.cuda.empty_cache()

# Load model and tokenizer
model_name = "KBTG-Labs/THaLLE-0.1-7B-fa"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../model-cache")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir="../model-cache", 
    use_cache=False,
    device_map=device,  
    torch_dtype=torch.bfloat16
)

# Load the data from CSV
df_articles = pd.read_csv('articles_cot.csv')
df_wangchan = pd.read_csv('wangchan.csv')

dataset_articles = Dataset.from_pandas(df_articles[['Instruction','Input', 'Output']])
dataset_wangchan = Dataset.from_pandas(df_wangchan[['Instruction','Input', 'Output']])
format_prompt = '''###Instruction: {}
###Input:
{}
###Output:
{}
'''

print('dataset articles:', dataset_articles)
print('dataset wangchan:', dataset_wangchan)

def chunks_tokenize_function(datas, context_length = 1024, overlap = 256):
    # Format the input text
    texts = [format_prompt.format(instuctions, inputs, outputs) 
             for instuctions, inputs, outputs in zip(datas['Instruction'],datas['Input'],datas['Output'])]
    # Create chunks of token IDs with overlap
    chunks_token_ids = {'input_ids': [], 'labels': [], 'length':[]}
    token_ids = list(map(lambda text : tokenizer.encode(text), texts))
    for instr, inp, token in zip(datas['Instruction'],datas['Input'],token_ids):
        token_length = len(token)
        if token_length < context_length:
            token += [tokenizer.eos_token_id] * (context_length - token_length)
            chunks_token_ids['input_ids'].append(token)
            chunks_token_ids['labels'].append(token)
            chunks_token_ids['length'].append(len(token))
            continue
        step = context_length - overlap
        loops = token_length / step
        loops = int(math.ceil(loops))
        # Split token_ids into chunks
        for i in range(loops):
            start = step * i
            end = min(start + context_length, len(token))
            chunk = token[start:end]
            chunk += [tokenizer.eos_token_id] * (context_length - len(chunk))
            # Append input_ids and labels (labels are the same as input_ids for causal LM)
            chunks_token_ids['input_ids'].append(chunk)
            chunks_token_ids['labels'].append(chunk)
            chunks_token_ids['length'].append(len(chunk))
    return chunks_token_ids

# Set up chunking parameters
# AVG Articles token length = 2367.2682119205297
# AVG Wangchan token length = 1046.9717277486911

# Articles dataset chunked and tokenized
tokenized_dataset = dataset_articles.map(
    lambda data: chunks_tokenize_function(data),
    batched=True,
    remove_columns=['Instruction', 'Input', 'Output']
)

# Train-test split
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.3)
print('Articles tokenized', tokenized_dataset)

# Wangchan dataset chunked and tokenized
tokenized_wangchan_dataset = dataset_wangchan.map(
    lambda data: chunks_tokenize_function(data),
    batched=True,
    remove_columns=['Instruction', 'Input', 'Output']
)

print('wangchan tokenized:',tokenized_wangchan_dataset)

# Concatenate training dataset and shuffle
tokenized_train_dataset = concatenate_datasets([tokenized_dataset['train'],tokenized_wangchan_dataset])
tokenized_train_dataset = tokenized_train_dataset.shuffle(seed=random.randint(0,50))
tokenized_dataset['train'] = tokenized_train_dataset

print('merged dataset',tokenized_dataset)

# Define LoRA configuration for fine-tuning
lora_config = LoraConfig(
    r=8,           
    lora_alpha=32, 
    target_modules=['q_proj', 'k_proj' ,'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], 
    lora_dropout=0.1,
    bias="none"    
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
model.print_trainable_parameters()

# Define training arguments for multi-GPU
training_args = TrainingArguments(
    output_dir='./results/lora_checkpoint_cot',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_steps=500,
    eval_steps=500,
    bf16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

# Start training
trainer.train()
trainer.evaluate()
# Save the trained LoRA model
model.save_pretrained('./lora_model_cot')

print('Training finished')