import sys
import os
import typing
import gc
import json
import logging
import traceback
import pandas as pd
import torch

# ============================================
# GPU & MEMORY OPTIMIZATION
# ============================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Helps prevent fragmentation by allowing segments to grow
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

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

# ============================================
# IMPORTS (Post-Patch)
# ============================================
import lm_eval
from lm_eval.utils import make_table

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('evaluation.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
CSV_PATH = r'/jumbo/yaoqingyang/ivprasad/EmbedLLM/data/model_order1.csv'
OUTPUT_DIR = "./results"
TASKS = ["mmlu", "truthfulqa_mc1", "piqa", "gsm8k", "gpqa"]

# Use "auto" to let lm-eval find the max batch size for your VRAM
BATCH_SIZE = 8 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def clear_gpu_memory():
    """Aggressive VRAM cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Synchronize ensures all kernels are finished before next model load
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=i)
    gc.collect()

def make_serializable(obj):
    if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    if hasattr(obj, 'item'): return obj.item()
    if hasattr(obj, 'tolist'): return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except:
        return str(obj)

# ============================================
# MAIN EVALUATION LOOP
# ============================================
models_df = pd.read_csv(CSV_PATH)
logger.info(f"Loaded {len(models_df)} models. Visible GPUs: {torch.cuda.device_count()}")

for i, row in models_df.iterrows():
    model_name = row['model_name']
    safe_name = model_name.replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
    
    if os.path.exists(output_path):
        logger.info(f"[{i+1}/{len(models_df)}] Skipping {model_name} (Results found).")
        continue
    
    clear_gpu_memory()
    logger.info(f"\n[{i+1}/{len(models_df)}] EVALUATING: {model_name}")
    
    # Model arguments configured for multi-GPU and memory efficiency
    # load_in_4bit=True is recommended for 7B+ models on limited VRAM
    model_args = (
        f"pretrained={model_name},"
        f"trust_remote_code=True,"
        f"device_map=auto,"
        f"load_in_4bit=True,"  # Change to load_in_8bit=True if you need higher precision
        f"max_length=4096"     # Limits KV cache size
    )
    
    try:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=TASKS,
            device=None,
            batch_size=BATCH_SIZE,
        )
        
        serializable_results = make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        logger.info(f"✅ Success: {model_name}")
        logger.info("\n" + make_table(results))
        
        del results, serializable_results
        
    except Exception as e:
        logger.error(f"❌ Error evaluating {model_name}")
        logger.error(traceback.format_exc())
        
        with open(os.path.join(OUTPUT_DIR, f"{safe_name}_ERROR.txt"), "w") as f:
            f.write(traceback.format_exc())
    
    finally:
        clear_gpu_memory()

logger.info("\n--- All Evaluations Finished ---")