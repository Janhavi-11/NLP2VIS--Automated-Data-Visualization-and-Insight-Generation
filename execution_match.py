import pandas as pd
import mysql.connector
import json
import os

def execute_query(connection, query):
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
    except Exception as e:
        print(f"❌ Query Execution Failed: {e}")
        result = None
    cursor.close()
    return result

def execution_match_accuracy(true_queries, pred_queries, db_connection):
    if len(true_queries) != len(pred_queries):
        raise ValueError("Length of true queries and predicted queries must be the same.")

    match_count = 0
    total = len(true_queries)

    for idx, (true_query, pred_query) in enumerate(zip(true_queries, pred_queries), start=1):
        true_result = execute_query(db_connection, true_query)
        pred_result = execute_query(db_connection, pred_query)

        if true_result is None or pred_result is None:
            print(f"⚠️ Skipping query pair {idx} due to execution failure.")
            continue

        true_df = pd.DataFrame(true_result)
        pred_df = pd.DataFrame(pred_result)

        try:
            true_df_sorted = true_df.sort_index(axis=1).sort_values(by=true_df.columns.tolist(), ignore_index=True)
            pred_df_sorted = pred_df.sort_index(axis=1).sort_values(by=pred_df.columns.tolist(), ignore_index=True)

            if true_df_sorted.equals(pred_df_sorted):
                match_count += 1
        except Exception as e:
            print(f"⚠️ Error comparing results for query pair {idx}: {e}")
            continue

    accuracy = (match_count / total) * 100
    return accuracy

if __name__ == "__main__":
    # Step 1: Load your dataset
    dataset_path = os.path.join("sql_graph_dashboard", "Remarks final dataset.csv")
    df = pd.read_csv(dataset_path)

    # Step 2: Extract the Actual and Predicted queries
    true_queries = df['Actual Queries'].dropna().tolist()
    pred_queries = df['Predicted Queries'].dropna().tolist()

    # Step 3: Load DB config
    with open(os.path.join("sql_graph_dashboard", "db_config.json")) as f:
        db_config = json.load(f)

    # Step 4: Connect to your database
    db_conn = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )

    # Step 5: Calculate Execution Match Accuracy
    accuracy = execution_match_accuracy(true_queries, pred_queries, db_conn)
    print(f"\n✅ Execution Match Accuracy: {accuracy:.2f}%")

    # Step 6: Close the database connection
    db_conn.close()
