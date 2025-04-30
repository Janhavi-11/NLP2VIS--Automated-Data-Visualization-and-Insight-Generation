import os
import re
import json
import torch
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import mysql.connector
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5_base_output import t5_base_output 

app = Flask(__name__)  

# Load T5 model and tokenizer
model_path = "t5base_sql_chart_model_finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load sentence transformer for OOC detection
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load training queries for OOC check
dataset_path = os.path.join("sql_graph_dashboard", "Major Project Final Dataset.csv")
df_train = pd.read_csv(dataset_path)
training_queries = df_train["NLP Query"].dropna().tolist()
training_sql_queries = df_train["SQL Query"].dropna().tolist()
training_chart_types = df_train["chart type"].dropna().tolist()

training_embeddings = embedder.encode(training_queries, convert_to_tensor=True)

# Load DB config
with open(os.path.join("sql_graph_dashboard", "db_config.json")) as f:
    db_config = json.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat')
def chat():
    return render_template("main.html")

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    from time import time
    user_query = request.json['query'].strip()

    if len(user_query.split()) <= 3:
        return jsonify({
            "error": "Invalid query: Please provide more details in your query."
        })

    fallback_used = False

    try:
        if is_out_of_context(user_query):
            return jsonify({
                "error": "Query is out of context. Please ask something related to the student database."
            })
        fallback_used = False
        try:
            model_output = get_model_output(user_query)
            sql_query, chart_type = parse_output(model_output)
        except Exception:
            matched_index = t5_base_output(user_query, training_queries)  # 🆕 Call function from separate file
            if matched_index is not None:
                sql_query = training_sql_queries[matched_index]
                chart_type = training_chart_types[matched_index]
                print("✅ Parsed SQL: ", sql_query)
                print("✅ Parsed Chart Type: ", chart_type)
            else:
                return jsonify({"error": "Failed to process your query."})

        db_result = run_sql_query(sql_query)

        if "error" in db_result:
            return jsonify({"error": db_result["error"]})

        df_result = pd.DataFrame(db_result["rows"], columns=db_result["columns"])
        chart_path = generate_chart_image(df_result, chart_type)

        if not chart_path:
            return jsonify({"error": "Chart data not suitable for plotting. Please try a different query."})

        return jsonify({
            "sql_query": sql_query,
            "chart_type": chart_type,
            "chart_url": f"/static/charts/{chart_path}?t={int(time())}",
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

def is_out_of_context(query, threshold=0.6):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, training_embeddings)[0]
    max_sim = float(torch.max(similarities))
    return max_sim < threshold

def get_model_output(user_input):
    input_text = "translate: " + user_input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(input_ids, max_length=128)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def parse_output(output):
    try:
        match = re.search(r"(?:<SQL>|SQL>)(.?)(?:<CHART>|CHART>)+(.)", output, re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()
            chart_type = match.group(2).strip()
        elif 'CHART>' in output:
            sql_query, chart_type = output.split('CHART>', 1)
            sql_query = sql_query.strip()
            chart_type = chart_type.strip()
        else:
            raise ValueError("Expected format: 'SQL_QUERY,CHART_TYPE'")
        
        if sql_query.upper().startswith('SQL>'):
            sql_query = sql_query[4:].strip()

        print("✅ Parsed SQL: ", sql_query)
        print("✅ Parsed Chart Type: ", chart_type)
        return sql_query, chart_type
    except Exception as e:
        raise ValueError(f"Failed to parse model output. Expected format: 'SQL_QUERY,CHART_TYPE'. Error: {str(e)}")

def run_sql_query(query):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = cursor.column_names
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": f"MySQL error: {str(e)}"}

def generate_chart_image(df, chart_type):
    import uuid
    plt.figure(figsize=(10, 6))
    chart_file = f"chart_{uuid.uuid4().hex}.png"
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, chart_file)

    try:
        if df.shape == (1, 1):
            value = df.iloc[0, 0]
            df = pd.DataFrame({"Metric": ["Result"], "Value": [value]})

        if df.shape[1] < 2:
            return None

        df[df.columns[1]] = pd.to_numeric(df[df.columns[1]], errors='coerce')

        if df[df.columns[1]].isnull().all():
            print("Chart generation error: No numeric data to plot")
            return None

        num_colors = len(df)
        colors = plt.colormaps['tab20']

        if chart_type.lower() == "bar chart":
            df.plot(kind='bar', x=df.columns[0], y=df.columns[1], color=[colors(i) for i in range(num_colors)], legend=False)
        elif chart_type.lower() == "line chart":
            df.plot(kind='line', x=df.columns[0], y=df.columns[1], marker='o', color=[colors(i) for i in range(num_colors)])
        elif chart_type.lower() == "pie chart":
            df.set_index(df.columns[0])[df.columns[1]].plot(kind='pie', autopct='%1.1f%%', colors=[colors(i) for i in range(num_colors)])
        elif chart_type.lower() == "scatter plot":
            df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], color=[colors(i) for i in range(num_colors)])
        else:
            return None

        plt.xticks(rotation=90, ha='left', fontsize=8)
        plt.tight_layout(pad=3.0)
        plt.title(chart_type)
        plt.savefig(chart_path)
        plt.close()
        return chart_file

    except Exception as e:
        print("Chart generation error:", e)
        return None

if __name__ == "__main__":  
    app.run(debug=True)