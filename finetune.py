import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pretrained model and tokenizer
model_path = r"E:\Major Project\nlp2vis\t5base_sql_chart_model"  # your base model path
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load your dataset
dataset = load_dataset("csv", data_files={
    "train": r"nlp2vis/sql_graph_dashboard/Major Project Final Dataset.csv"
})

# Table schema (inject into every prompt)
table_schema_prompt = """
Tables:
- student_admss_info(pid, acadyear, stat, month_year, gender)
- college_students_primary_info(pid, Name, branch)
- students_result_info(pid, sem, gpa, result)
- students_prev_edu(pid, hsc, cet, diploma)
- students_hobby_info(pid, ECategory, Hobby, Hobby_level)
- students_placement(pid, Placed, Package)
"""

# Preprocess function
def preprocess_data(example):
    input_text = table_schema_prompt.strip() + "\n\ntranslate: " + example["NLP Query"]
    target_text = f"{example['SQL Query'].strip()}, {example['chart type'].strip()}"
    
    model_inputs = tokenizer(
        input_text, max_length=512, padding="max_length", truncation=True
    )
    labels = tokenizer(
        target_text, max_length=128, padding="max_length", truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_data, remove_columns=dataset["train"].column_names)

# Training setup
training_args = TrainingArguments(
    output_dir="./t5base_sql_chart_model_finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=4,  # Try increasing for better results
    logging_dir="./logs_finetune",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="no",  # You can enable eval later
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train
trainer.train()

# Save the finetuned model
model.save_pretrained("./t5base_sql_chart_model_finetuned")
tokenizer.save_pretrained("./t5base_sql_chart_model_finetuned")

print("✅ Finetuning complete. Model saved to './t5base_sql_chart_model_finetuned'")
