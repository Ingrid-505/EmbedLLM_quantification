import json
import os
import pandas as pd

def extract_correctness_from_harness(results_folder):
    # Hold {model_name: {question_id: 0/1}}
    master_data = {}
    all_question_ids = set()

    for file_name in os.listdir(results_folder):
        if file_name.endswith(".json"):
            model_name = file_name.replace("_results.json", "")
            with open(os.path.join(results_folder, file_name), 'r') as f:
                data = json.load(f)
            
            model_results = {}
            # 'results' in lm-harness logs contains the per-task accuracy
            # 'samples' contains individual correctness
            for task, samples in data.get("samples", {}).items():
                for idx, sample in enumerate(samples):
                    # Create a unique ID for each question across tasks
                    q_id = f"{task}_{idx}" 
                    # The harness uses 'acc' or 'exact_match' as the binary label
                    is_correct = 1 if sample.get("acc") or sample.get("exact_match") else 0
                    
                    model_results[q_id] = is_correct
                    all_question_ids.add(q_id)
            
            master_data[model_name] = model_results

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(master_data, orient='index')
    # Fill any missing values with 0 (assuming model failed if no result exists)
    df = df.fillna(0).astype(int)
    
    return df

results_path = "./results"
matrix = extract_correctness_from_harness(results_path)

matrix.to_csv("train_correctness.csv")
pd.DataFrame(matrix.index).to_csv("model_order.csv", index=False, header=False)
pd.DataFrame(matrix.columns).to_csv("question_order.csv", index=False, header=False)

print(f"Matrix created with {matrix.shape[0]} models and {matrix.shape[1]} questions.")