import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# Detect device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_path = r"E:\Major Project\nlp2vis\t5base_sql_chart_model_finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# Load test dataset
test_file_path = r"E:\Major Project\nlp2vis\sql_graph_dashboard\test_dataset.csv"
df = pd.read_csv(test_file_path)

# Validate input column
if "NLP Query" not in df.columns:
    raise ValueError(f"Expected column 'NLP Query' not found. Found: {df.columns}")

# Inference parameters
batch_size = 16
max_input_len = 128
max_output_len = 128

# Store results
sql_outputs = []
chart_outputs = []

# Run inference in batches
queries = df["NLP Query"].tolist()
for i in tqdm(range(0, len(queries), batch_size), desc="Generating predictions"):
    batch = queries[i:i + batch_size]
    inputs = ["translate: " + q for q in batch]
    tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            max_length=max_output_len
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for out in decoded:
        sql_output, chart_output = "", ""
        if "<CHART>" in out:
            parts = out.split("<CHART>")
            sql_output = parts[0].replace("<SQL>", "").replace("SQL>", "").strip()
            chart_output = parts[1].replace("CHART>", "").strip()
        else:
            sql_output = out.replace("<SQL>", "").replace("SQL>", "").strip()

        sql_outputs.append(sql_output)
        chart_outputs.append(chart_output)

# Save outputs
df["SQL Query"] = sql_outputs
df["chart type"] = chart_outputs
output_path = r"E:\Major Project\nlp2vis\sql_graph_dashboard\model_output_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to: {output_path}")
