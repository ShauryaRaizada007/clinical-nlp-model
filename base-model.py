# Base model training pipeline for doctor-patient conversation modeling

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import json

"""# Load train split
with open("C:\\Users\\HP\\Downloads\\english-train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
print("Train size:", len(train_data))

print(train_data[0])"""

# Step 1: Load pre-trained base model (lightweight for fine-tuning)
model_name = "microsoft/phi-2"  # You can switch to TinyLLaMA or Mistral later
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Load and prepare dataset (e.g., MedDialog processed.en)
with open("C:\\Users\\HP\\Downloads\\english-train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

examples = []
for entry in data:
    dialogue = "\n".join(entry["utterances"])
    prompt = (
        "### Instruction:\n"
        "Summarize the diagnosis, medication, and recommendation from the conversation below.\n\n"
        f"### Conversation:\n{dialogue}\n\n"
        "### Response:\n"
    )
    examples.append({"prompt": prompt, "completion": ""})  # You can add GPT-generated completions here

# Convert to HuggingFace dataset
hf_dataset = Dataset.from_list(examples)

# Step 3: Tokenize

def tokenize(example):
    text = example["prompt"] + example["completion"]
    return tokenizer(text, padding="max_length", truncation=True, max_length=1024)

tokenized_dataset = hf_dataset.map(tokenize, batched=False)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./clinical-base-model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 5: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 6: Train
trainer.train()
