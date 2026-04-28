import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Check and assign the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset
csv_file = "D:\\VPS\\questions.csv"  # Update with your actual file path
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} questions from {csv_file}")

# Display the first few rows to verify structure
print(df.head())

# Ensure the CSV file has the required columns
if "Question" not in df.columns or "Label" not in df.columns:
    raise ValueError("The CSV file must contain 'Question' and 'Label' columns.")

# Split the dataset into training and evaluation sets (80-20 split)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Load ClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["Question"], padding="max_length", truncation=True, max_length=128)

# Tokenize the datasets
print("Tokenizing the datasets...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
eval_tokenized = eval_dataset.map(tokenize_function, batched=True)

# Add labels to tokenized datasets
train_tokenized = train_tokenized.map(lambda x: {"labels": x["Label"]})
eval_tokenized = eval_tokenized.map(lambda x: {"labels": x["Label"]})

# Training arguments
training_args = TrainingArguments(
    output_dir="./clinicalbert_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
)

# Compute accuracy metric
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
print("Training ClinicalBERT...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./clinicalbert_finetuned")
tokenizer.save_pretrained("./clinicalbert_finetuned")
print("ClinicalBERT fine-tuned and saved!")
