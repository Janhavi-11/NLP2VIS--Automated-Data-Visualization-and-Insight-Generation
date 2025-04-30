import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load your dataset (change this path to your actual file)
dataset = load_dataset("csv", data_files={
    "train": r"C:\Users\B15\Downloads\Major-Project-sem-8-main\Major Proj dataset.csv"
})

# Check column names
print("Columns:", dataset["train"].column_names)

# Function to format both SQL + chart in output
def preprocess_data(example):
    # Prepare input and target texts
    input_text = "translate: " + example["NLP Query"]
    target_text = f"<SQL> {example['SQL Query']} <CHART> {example['chart type']}"
    
    # Tokenize input and output
    model_input = tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    label = tokenizer(
        text=target_text,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    model_input["labels"] = label["input_ids"]
    return model_input

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_data)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./t5base_sql_chart_model",  # where to save the model and logs
    per_device_train_batch_size=4,         # batch size per device during training
    num_train_epochs=5,                    # number of training epochs
    logging_dir="./logs",                  # directory to store logs
    logging_steps=10,                      # log every 10 steps
    save_strategy="epoch",                 # save model at the end of each epoch
    evaluation_strategy="no",              # no evaluation during training (you can set "epoch" for evaluation per epoch)
    learning_rate=3e-5,                    # learning rate
    weight_decay=0.01,                     # weight decay for regularization
)

# Initialize Trainer
trainer = Trainer(
    model=model,                           # the model to be trained
    args=training_args,                    # the training arguments
    train_dataset=dataset["train"],        # the training dataset
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./t5base_sql_chart_model")
tokenizer.save_pretrained("./t5base_sql_chart_model")

print("✅ Training complete. Model saved.")