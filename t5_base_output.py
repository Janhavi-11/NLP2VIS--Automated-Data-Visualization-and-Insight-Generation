def t5_base_output(user_query, training_queries):
    try:
        if user_query in training_queries:
            matched_index = training_queries.index(user_query)
            return matched_index
        return None
    except Exception as e:
        return jsonify({"error": "Failed to process your query."})