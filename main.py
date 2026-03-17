from datetime import datetime
import os
import warnings

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Suppress warnings
warnings.filterwarnings("ignore")

# Imports
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoModel
)
from datasets import load_dataset
from xai import evaluateXAI
import numpy as np
import pandas as pd
import evaluate
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    # 1. Tokenize the text
    outputs = tokenizer(example["review_text"], truncation=True, padding="max_length", max_length=256)
    
    # 2. Map 'is_spoiler' to 'labels' and ensure they are integers
    # BERT expects labels to be 0 or 1 integers for classification
    outputs["labels"] = [int(label) for label in example["is_spoiler"]]
    
    return outputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "rmse": rmse
    }

def spoilerChecker():     
    dataset = load_dataset("json", data_files="imdbDataset/IMDB_reviews.json")

    num_samples = 20000
    # num_samples = int(len(dataset["train"])  * 0.4)
    test_size = 16000
    # test_size = len(dataset["train"]) - int(len(dataset["train"])  * 0.2)
    train_subset = dataset["train"].shuffle(seed=42).select(range(num_samples))
    # working on a very small sample, just to present the process

    tokenized_dataset = train_subset.map(tokenize_fn, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_dataset.features

    model_path = "./my_spoiler_model"

    if os.path.exists(model_path):
        print("--- Loading existing model from disk ---")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        print("--- No model found. Starting training... ---")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        training_args = TrainingArguments(
            output_dir="./bert-imdb",
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            save_strategy="epoch",
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset = tokenized_dataset.select(range(test_size)),
            eval_dataset = tokenized_dataset.select(range(test_size, num_samples)),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(model_path)

        trainer.evaluate()
    evaluateXAI(dataset)

    # 0. Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Move the model to the device (do this once)
    model.to(device)
    model.eval() # Set to evaluation mode

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File pattern: YYYY_MM_DD_spoilerFeedback.txt
    log_filename = f"{datetime.now().strftime('%Y_%m_%d')}_spoilerFeedback.txt"
    log_path = os.path.join(log_dir, log_filename)

    print("\n" + "="*30)
    print("SPOILER DETECTOR READY")
    print("Type 'quit' or 'exit' to stop.")
    print("="*30)

    while True:
        user_input = input("\nEnter a movie review text: ")
        
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue

        # 1. Tokenize and move to device
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(device) # <--- Moves all tensors (input_ids, mask) to GPU/CPU

        # 2. Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        # 4. Result Formatting
        if predicted_class == 1:
            result_text = "🛑 RESULT: Warning! This looks like a SPOILER."
        else:
            result_text = "✅ RESULT: This review seems safe."

        print(result_text)

        # 5. Save to Log File
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Message: \"{user_input}\"\n")
            f.write(f"{result_text}\n")
            f.write("-" * 20 + "\n")

if __name__ == "__main__":
    spoilerChecker()