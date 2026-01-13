# # # import pandas as pd
# # # import subprocess
# # # import os

# # # # 1. Configuration
# # # CSV_PATH = r'/jumbo/yaoqingyang/ivprasad/EmbedLLM/data/model_order.csv'
# # # OUTPUT_DIR = "./results"
# # # TASKS = "mmlu,truthfulqa_mc1,social_iqa,piqa,medmcqa,mathqa,logiqa,gsm8k,gpqa,asdiv"
# # # BATCH_SIZE = "auto"
# # # DEVICE = "cuda:4"

# # # os.makedirs(OUTPUT_DIR, exist_ok=True)
# # # os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# # # # 2. Load Models
# # # df = pd.read_csv(CSV_PATH)

# # # for index, row in df.iterrows():
# # #     model_name = row['model_name']
# # #     safe_name = model_name.replace("/", "_")
# # #     output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
    
# # #     if os.path.exists(output_path):
# # #         print(f"Skipping {model_name}, results already exist.")
# # #         continue

# # #     print(f"\n[{index+1}/{len(df)}] EVALUATING: {model_name}")
    
# # #     # 3. Construct the Bash Command
# # #     command = [
# # #         "lm_eval",
# # #         "--model", "hf",
# # #         "--model_args", f"pretrained={model_name},trust_remote_code=True",
# # #         "--tasks", TASKS,
# # #         "--device", DEVICE,
# # #         "--batch_size", BATCH_SIZE,
# # #         "--output_path", output_path,
# # #         "--write_out"
# # #     ]
    
# # #     try:
# # #         subprocess.run(command, check=True)
# # #         print(f"Finished {model_name}. Results saved to {output_path}")
# # #     except subprocess.CalledProcessError as e:
# # #         print(f"Error evaluating {model_name}: {e}")
# # #         continue

# # # print("\nAll evaluations completed!")
# # import os
# # import sys
# # import json
# # import pandas as pd
# # import dataclasses

# # # --- ENVIRONMENT PATCHES ---
# # os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# # # Patch for Python 3.12 Dataclass TypeError
# # original_fields = dataclasses.fields
# # def patched_fields(class_or_instance):
# #     try:
# #         return original_fields(class_or_instance)
# #     except TypeError:
# #         return []
# # dataclasses.fields = patched_fields

# # # Import lm_eval AFTER the patch
# # import lm_eval
# # from lm_eval.utils import make_table

# # # --- CONFIGURATION ---
# # CSV_PATH = r'/jumbo/yaoqingyang/ivprasad/EmbedLLM/data/model_order.csv'
# # OUTPUT_DIR = "./results"
# # # FIX 1: Tasks should be a list, not a comma-separated string
# # TASKS = ["mmlu", "truthfulqa_mc1", "social_iqa", "piqa", "medmcqa", 
# #          "mathqa", "logiqa", "gsm8k", "gpqa", "asdiv"]
# # BATCH_SIZE = "auto"
# # DEVICE = "cuda:7"

# # os.makedirs(OUTPUT_DIR, exist_ok=True)
# # models_df = pd.read_csv(CSV_PATH)

# # print(f"--- Research Pipeline: API Mode (Python {sys.version.split()[0]}) ---")

# # # FIX 2: Helper function to make results JSON-serializable
# # def make_serializable(obj):
# #     """Convert non-serializable objects to serializable types."""
# #     if isinstance(obj, dict):
# #         return {k: make_serializable(v) for k, v in obj.items()}
# #     elif isinstance(obj, list):
# #         return [make_serializable(v) for v in obj]
# #     elif hasattr(obj, 'item'):  # numpy/torch scalar
# #         return obj.item()
# #     elif hasattr(obj, 'tolist'):  # numpy/torch array
# #         return obj.tolist()
# #     else:
# #         try:
# #             json.dumps(obj)
# #             return obj
# #         except (TypeError, ValueError):
# #             return str(obj)

# # for i, row in models_df.iterrows():
# #     model_name = row['model_name']
# #     safe_name = model_name.replace("/", "_")
# #     output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
    
# #     if os.path.exists(output_path):
# #         print(f"[{i+1}/{len(models_df)}] Skipping {model_name} (Results found).")
# #         continue
        
# #     print(f"\n[{i+1}/{len(models_df)}] EVALUATING: {model_name}")
    
# #     import logging
# #     import traceback

# #     # Set up logging at the top of your script
# #     logging.basicConfig(
# #         level=logging.DEBUG,
# #         format='%(asctime)s - %(levelname)s - %(message)s',
# #         handlers=[
# #             logging.FileHandler('evaluation.log'),
# #             logging.StreamHandler(sys.stdout)
# #         ]
# #     )
# #     logger = logging.getLogger(__name__)

# #     # Then in your loop:
# #     for i, row in models_df.iterrows():
# #         model_name = row['model_name']
# #         safe_name = model_name.replace("/", "_")
# #         output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
        
# #         if os.path.exists(output_path):
# #             logger.info(f"[{i+1}/{len(models_df)}] Skipping {model_name} (Results found).")
# #             continue
            
# #         logger.info(f"\n[{i+1}/{len(models_df)}] EVALUATING: {model_name}")
        
# #         try:
# #             results = lm_eval.simple_evaluate(
# #                 model="hf",
# #                 model_args=f"pretrained={model_name},trust_remote_code=True",
# #                 tasks=TASKS,
# #                 device=DEVICE,
# #                 batch_size=BATCH_SIZE,
# #             )
            
# #             serializable_results = make_serializable(results)
            
# #             with open(output_path, "w") as f:
# #                 json.dump(serializable_results, f, indent=2, default=str)
                
# #             logger.info(f"✅ Success: {model_name}")
# #             logger.info(make_table(results))
            
# #         except Exception as e:
# #             logger.error(f"❌ Error evaluating {model_name}")
# #             logger.error(f"Exception type: {type(e).__name__}")
# #             logger.error(f"Exception message: {str(e)}")
# #             logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
# #             # Also save error to a separate file for this model
# #             error_path = os.path.join(OUTPUT_DIR, f"{safe_name}_ERROR.txt")
# #             with open(error_path, "w") as f:
# #                 f.write(f"Model: {model_name}\n")
# #                 f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
# #                 f.write(traceback.format_exc())
            
# #             continue

# # print("\n--- All Evaluations Finished ---")
# import sys
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
# import typing

# # ============================================
# # PYTHON 3.12 COMPATIBILITY PATCHES
# # Must be at the VERY TOP before other imports
# # ============================================

# # Patch 1: Fix typing.List instantiation issue
# if not hasattr(typing, '_GenericAlias_patched'):
#     _original_generic_alias_call = typing._GenericAlias.__call__
    
#     def _patched_generic_alias_call(self, *args, **kwargs):
#         origin = getattr(self, '__origin__', None)
#         if origin is list:
#             return list(*args, **kwargs)
#         elif origin is dict:
#             return dict(*args, **kwargs)
#         elif origin is set:
#             return set(*args, **kwargs)
#         elif origin is tuple:
#             return tuple(*args, **kwargs)
#         elif origin is frozenset:
#             return frozenset(*args, **kwargs)
#         return _original_generic_alias_call(self, *args, **kwargs)
    
#     typing._GenericAlias.__call__ = _patched_generic_alias_call
#     typing._GenericAlias_patched = True

# # Patch 2: Fix dataclasses.fields issue
# import dataclasses
# original_fields = dataclasses.fields
# def patched_fields(class_or_instance):
#     try:
#         return original_fields(class_or_instance)
#     except TypeError:
#         return []
# dataclasses.fields = patched_fields

# # ============================================
# # NOW IMPORT EVERYTHING ELSE
# # ============================================
# import os
# import json
# import logging
# import traceback
# import pandas as pd

# os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# import lm_eval
# from lm_eval.utils import make_table

# # ============================================
# # LOGGING SETUP
# # ============================================
# logging.basicConfig(
#     level=logging.INFO,  # Changed from DEBUG to reduce noise
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('evaluation.log'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # Suppress noisy debug logs from other libraries
# logging.getLogger("filelock").setLevel(logging.WARNING)
# logging.getLogger("datasets").setLevel(logging.WARNING)
# logging.getLogger("transformers").setLevel(logging.WARNING)

# # ============================================
# # CONFIGURATION
# # ============================================
# CSV_PATH = r'/jumbo/yaoqingyang/ivprasad/EmbedLLM/data/model_order.csv'
# OUTPUT_DIR = "./results"
# # TASKS = ["mmlu", "truthfulqa_mc1", "social_iqa", "piqa", "medmcqa", 
# #          "mathqa", "logiqa", "gsm8k", "gpqa", "asdiv"]
# TASKS = [
#     "mmlu",
#     "truthfulqa_mc1",
#     "piqa",
#     "gsm8k",
#     "gpqa",
#     # "social_iqa",  # BROKEN - uses .py script
#     # "medmcqa",     # May be broken
#     # "mathqa",      # BROKEN - uses .py script  
#     # "logiqa",      # May be broken
#     # "asdiv",       # May be broken
# ]
# BATCH_SIZE = "auto"
# DEVICE = "cuda:4,5,6"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ============================================
# # HELPER FUNCTIONS
# # ============================================
# def make_serializable(obj):
#     """Convert non-serializable objects to serializable types."""
#     if isinstance(obj, dict):
#         return {k: make_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [make_serializable(v) for v in obj]
#     elif hasattr(obj, 'item'):
#         return obj.item()
#     elif hasattr(obj, 'tolist'):
#         return obj.tolist()
#     else:
#         try:
#             json.dumps(obj)
#             return obj
#         except (TypeError, ValueError):
#             return str(obj)

# # ============================================
# # MAIN EVALUATION LOOP
# # ============================================
# models_df = pd.read_csv(CSV_PATH)
# logger.info(f"--- Research Pipeline: Python {sys.version.split()[0]} ---")
# logger.info(f"Loaded {len(models_df)} models from {CSV_PATH}")

# for i, row in models_df.iterrows():
#     model_name = row['model_name']
#     safe_name = model_name.replace("/", "_")
#     output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
    
#     if os.path.exists(output_path):
#         logger.info(f"[{i+1}/{len(models_df)}] Skipping {model_name} (Results found).")
#         continue
        
#     logger.info(f"\n[{i+1}/{len(models_df)}] EVALUATING: {model_name}")
    
#     try:
#         results = lm_eval.simple_evaluate(
#             model="hf",
#             model_args=f"pretrained={model_name},trust_remote_code=True",
#             tasks=TASKS,
#             device=DEVICE,
#             batch_size=BATCH_SIZE,
#         )
        
#         serializable_results = make_serializable(results)
        
#         with open(output_path, "w") as f:
#             json.dump(serializable_results, f, indent=2, default=str)
            
#         logger.info(f"✅ Success: {model_name}")
#         logger.info("\n" + make_table(results))
        
#     except Exception as e:
#         logger.error(f"❌ Error evaluating {model_name}")
#         logger.error(f"Exception: {type(e).__name__}: {str(e)}")
#         logger.error(f"Traceback:\n{traceback.format_exc()}")
        
#         error_path = os.path.join(OUTPUT_DIR, f"{safe_name}_ERROR.txt")
#         with open(error_path, "w") as f:
#             f.write(f"Model: {model_name}\n")
#             f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
#             f.write(traceback.format_exc())
        
#         continue

# logger.info("\n--- All Evaluations Finished ---")
import sys
import os
import typing

# ============================================
# USE ALL AVAILABLE GPUs
# ============================================
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # Add more GPUs here

# ============================================
# PYTHON 3.12 COMPATIBILITY PATCHES
# ============================================
if not hasattr(typing, '_GenericAlias_patched'):
    _original_generic_alias_call = typing._GenericAlias.__call__
    
    def _patched_generic_alias_call(self, *args, **kwargs):
        origin = getattr(self, '__origin__', None)
        if origin is list:
            return list(*args, **kwargs)
        elif origin is dict:
            return dict(*args, **kwargs)
        elif origin is set:
            return set(*args, **kwargs)
        elif origin is tuple:
            return tuple(*args, **kwargs)
        elif origin is frozenset:
            return frozenset(*args, **kwargs)
        return _original_generic_alias_call(self, *args, **kwargs)
    
    typing._GenericAlias.__call__ = _patched_generic_alias_call
    typing._GenericAlias_patched = True

import dataclasses
original_fields = dataclasses.fields
def patched_fields(class_or_instance):
    try:
        return original_fields(class_or_instance)
    except TypeError:
        return []
dataclasses.fields = patched_fields

# ============================================
# IMPORTS
# ============================================
import gc
import json
import logging
import traceback
import pandas as pd
import torch

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import lm_eval
from lm_eval.utils import make_table

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ============================================
# CONFIGURATION
# ============================================
CSV_PATH = r'/jumbo/yaoqingyang/ivprasad/EmbedLLM/data/model_order.csv'
OUTPUT_DIR = "./results"

TASKS = [
    "mmlu",
    "truthfulqa_mc1",
    "piqa",
    "gsm8k",
    "gpqa",
]

BATCH_SIZE = "auto"  # Keep auto

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ============================================
# VERIFY GPU SETUP
# ============================================
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

# ============================================
# MAIN EVALUATION LOOP
# ============================================
models_df = pd.read_csv(CSV_PATH)
logger.info(f"Loaded {len(models_df)} models")

for i, row in models_df.iterrows():
    model_name = row['model_name']
    safe_name = model_name.replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_results.json")
    
    if os.path.exists(output_path):
        logger.info(f"[{i+1}/{len(models_df)}] Skipping {model_name} (Results found).")
        continue
    
    clear_gpu_memory()
    logger.info(f"\n[{i+1}/{len(models_df)}] EVALUATING: {model_name}")
    
    try:
        # KEY: device_map=auto spreads model across all visible GPUs
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},trust_remote_code=True,device_map=auto",
            tasks=TASKS,
            device=None,  # MUST be None with device_map
            batch_size=BATCH_SIZE,
        )
        
        serializable_results = make_serializable(results)
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        logger.info(f"✅ Success: {model_name}")
        logger.info("\n" + make_table(results))
        
        del results, serializable_results
        
    except Exception as e:
        logger.error(f"❌ Error: {model_name}")
        logger.error(f"{type(e).__name__}: {str(e)}")
        
        error_path = os.path.join(OUTPUT_DIR, f"{safe_name}_ERROR.txt")
        with open(error_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
            f.write(traceback.format_exc())
    
    finally:
        clear_gpu_memory()

logger.info("\n--- All Evaluations Finished ---")