import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub import HfApi, Repository

# Load dataset
dataset_path = 'data.csv'  # Update this with the actual path to your dataset
data = pd.read_csv(dataset_path)

# Prepare dataset
def preprocess_data(examples):
    return {
        'input_text': examples['UNT user input'],
        'output_text': examples['text']
    }

# Convert DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(data)
dataset = dataset.map(preprocess_data)

# Tokenizer and Model
model_name = 'openchat/openchat_3.5'  # Replace with the model you want to fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['input_text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=tokenized_dataset['test']  # if you have a test set
)

# Train
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Hugging Face Hub Integration
hf_username = 'omar07ib'  # Replace with your Hugging Face username
model_id = 'unt-7b'  # Replace with your model name
hf_token = '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'  # Replace with your Hugging Face token

# Authenticate and create a new repository
api = HfApi()
api.create_repo(token=hf_token, name=model_id, exist_ok=True)
repo_url = api.upload_file(
    token=hf_token,
    path_or_fileobj='./fine_tuned_model/pytorch_model.bin',
    path_in_repo='pytorch_model.bin',
    repo_id=f"{hf_username}/{model_id}"
)

print(f"Model successfully uploaded to Hugging Face Hub at: {repo_url}")
