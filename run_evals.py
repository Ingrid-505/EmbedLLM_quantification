import pandas as pd
import subprocess
import os

# 1. Configuration
CSV_PATH = r'/jumbo/yaoqingyang/ivprasad/EmbedLLM/data/model_order.csv'
OUTPUT_DIR = "./results"
TASKS = "mmlu,truthfulqa_mc1,social_iqa,piqa,medmcqa,mathqa,logiqa,gsm8k,gpqa,asdiv"
BATCH_SIZE = "auto"
DEVICE = "cuda:0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Models
df = pd.read_csv(CSV_PATH)

for index, row in df.iterrows():
    model_name = row['model_name']
    safe_name = model_name.replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
    
    if os.path.exists(output_path):
        print(f"Skipping {model_name}, results already exist.")
        continue

    print(f"\n[{index+1}/{len(df)}] EVALUATING: {model_name}")
    
    # 3. Construct the Bash Command
    command = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},trust_remote_code=True",
        "--tasks", TASKS,
        "--device", DEVICE,
        "--batch_size", BATCH_SIZE,
        "--output_path", output_path,
        "--write_out"
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Finished {model_name}. Results saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_name}: {e}")
        continue

print("\nAll evaluations completed!")