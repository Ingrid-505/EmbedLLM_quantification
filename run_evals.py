import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,5,7"
import json
import torch
import typing
import pandas as pd

# ============================================
# PYTHON 3.12 COMPATIBILITY PATCHES
# ============================================
if not hasattr(typing, '_GenericAlias_patched'):
    _original_generic_alias_call = typing._GenericAlias.__call__
    def _patched_generic_alias_call(self, *args, **kwargs):
        origin = getattr(self, '__origin__', None)
        if origin in [list, dict, set, tuple, frozenset]:
            return origin(*args, **kwargs)
        return _original_generic_alias_call(self, *args, **kwargs)
    typing._GenericAlias.__call__ = _patched_generic_alias_call
    typing._GenericAlias_patched = True

import dataclasses
original_fields = dataclasses.fields
def patched_fields(class_or_instance):
    try: return original_fields(class_or_instance)
    except TypeError: return []
dataclasses.fields = patched_fields

import lm_eval
from lm_eval.utils import make_table

# --- CONFIGURATION ---
INPUT_CSV = 'wuyang_models_ishan.csv' 
OUTPUT_DIR = './results1'
TASKS = ["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc1"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(INPUT_CSV)
    models = df['model_name'].tolist()

    for model_name in models:
        # Standard filename: swaps '/' for '--' to avoid path errors
        safe_name = model_name.replace('/', '--')
        result_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")

        if os.path.exists(result_path):
            continue

        print(f"Running: {model_name}")

        try:
            # DIRECT EVALUATION
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_name},trust_remote_code=True",
                tasks=TASKS,
                # device="cuda:0",
                batch_size="auto"
            )

            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)

            print(make_table(results))

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()