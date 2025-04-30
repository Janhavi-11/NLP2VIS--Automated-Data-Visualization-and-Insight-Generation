import pandas as pd

# Load ground truth and prediction files
gt_path = r"E:\Major Project\nlp2vis\sql_graph_dashboard\test_dataset[1].csv"
pred_path = r"E:\Major Project\nlp2vis\sql_graph_dashboard\model_output_cleaned.csv"

df_gt = pd.read_csv(gt_path)
df_pred = pd.read_csv(pred_path)

# Merge on NLP Query
df = pd.merge(df_gt, df_pred, on="NLP Query", suffixes=("_GT", "_PRED"))

# Extract columns
true_sqls = df["SQL Query_GT"]
pred_sqls = df["SQL Query_PRED"]
true_charts = df["chart type_GT"]
pred_charts = df["chart type_PRED"]

# Exact Match Accuracy
def exact_match(pred_list, true_list):
    return sum([str(p).strip() == str(t).strip() for p, t in zip(pred_list, true_list)]) / len(true_list)

# Logical Match Accuracy (case-insensitive, stripped)
def logical_match(pred_list, true_list):
    return sum([str(p).strip().lower() == str(t).strip().lower() for p, t in zip(pred_list, true_list)]) / len(true_list)

# Token-level Precision, Recall, F1 using sets
def calc_token_metrics(pred_list, true_list):
    tp = 0  # true positives (correct tokens)
    fp = 0  # false positives (extra tokens)
    fn = 0  # false negatives (missing tokens)

    for pred, true in zip(pred_list, true_list):
        pred_tokens = set(str(pred).lower().split())
        true_tokens = set(str(true).lower().split())

        tp += len(pred_tokens & true_tokens)
        fp += len(pred_tokens - true_tokens)
        fn += len(true_tokens - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# SQL Evaluation
sql_exact = exact_match(pred_sqls, true_sqls)
sql_logical = logical_match(pred_sqls, true_sqls)
sql_precision, sql_recall, sql_f1 = calc_token_metrics(pred_sqls, true_sqls)

# Chart Type Evaluation
chart_exact = exact_match(pred_charts, true_charts)
chart_logical = logical_match(pred_charts, true_charts)
chart_precision, chart_recall, chart_f1 = calc_token_metrics(pred_charts, true_charts)

# Print results
print("\n=== SQL Query Evaluation ===")
print(f"   Exact Match Accuracy     : {sql_exact:.2%}")
print(f"   Logical Match Accuracy   : {sql_logical:.2%}")
print(f"   Token-level Precision    : {sql_precision:.2%}")
print(f"   Token-level Recall       : {sql_recall:.2%}")
print(f"   Token-level F1 Score     : {sql_f1:.2%}")

print("\n=== Chart Type Evaluation ===")
print(f"   Exact Match Accuracy     : {chart_exact:.2%}")
print(f"   Logical Match Accuracy   : {chart_logical:.2%}")
print(f"   Token-level Precision    : {chart_precision:.2%}")
print(f"   Token-level Recall       : {chart_recall:.2%}")
print(f"   Token-level F1 Score     : {chart_f1:.2%}")