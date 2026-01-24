import os
import json
import argparse
import traceback
import torch
import typing
import dataclasses


if not hasattr(typing, "_GenericAlias_patched"):
    _original_generic_alias_call = typing._GenericAlias.__call__

    def _patched_generic_alias_call(self, *args, **kwargs):
        origin = getattr(self, "__origin__", None)
        if origin in [list, dict, set, tuple, frozenset]:
            return origin(*args, **kwargs)
        return _original_generic_alias_call(self, *args, **kwargs)

    typing._GenericAlias.__call__ = _patched_generic_alias_call
    typing._GenericAlias_patched = True

original_fields = dataclasses.fields
def patched_fields(class_or_instance):
    try:
        return original_fields(class_or_instance)
    except TypeError:
        return []
dataclasses.fields = patched_fields

import lm_eval
from lm_eval.utils import make_table

DEFAULT_TASKS = ["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc1"]

def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")

def build_model_args(model_name: str, extra: str = "") -> str:
    # IMPORTANT:
    # - No commas inside values (lm-eval splits by comma).
    # - Rely on CUDA_VISIBLE_DEVICES externally to choose 1/2/3 GPUs.
    base = [
        f"pretrained={model_name}",
        "trust_remote_code=True",
        "dtype=float16",
        "low_cpu_mem_usage=True",
        "device_map=auto",
        "parallelize=True",  # if your lm-eval ignores it, device_map still helps
    ]
    if extra:
        # user can pass: "load_in_8bit=True" etc.
        base.append(extra)
    return ",".join(base)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=2)  # keep stable for now
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--model_args_extra", type=str, default="")  # optional extra key=val (no commas)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name
    safe_name = safe_model_name(model_name)
    result_path = os.path.join(args.output_dir, f"{safe_name}.json")
    err_path = os.path.join(args.output_dir, f"{safe_name}.error.txt")

    if os.path.exists(result_path):
        print(f"[SKIP] Exists: {result_path}")
        return

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_args = build_model_args(model_name, extra=args.model_args_extra)

    print(f"[RUN] model={model_name}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"[INFO] tasks={tasks} batch_size={args.batch_size}")
    print(f"[INFO] model_args={model_args}")

    try:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size=args.batch_size,
        )
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        print(make_table(results))

        # remove old error file if any
        if os.path.exists(err_path):
            os.remove(err_path)

    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] evaluating {model_name}\n{tb}")
        with open(err_path, "w") as f:
            f.write(tb)

    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
