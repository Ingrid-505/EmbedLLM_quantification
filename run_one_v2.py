import os
import json
import time
import argparse
import traceback
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.utils import make_table



def net_esd_estimator(
            net,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers='xmin_mid',
            xmin_pos=2,
            conv_norm=0.5,
            filter_zeros=True,
            model_name=None,
            save_dir=None,
):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = OrderedDict({
        'D': [],
        'M': [],
        'N': [],
        'alpha': [],
        'alpha_weighted': [],
        'entropy': [],
        'log_alpha_norm': [],
        'log_norm': [],
        'log_spectral_norm': [],
        'longname': [],
        'matrix_rank': [],
        'norm': [],
        'num_evals': [],
        'stable_rank': [],
        'xmax': [],
        'xmin': [],
        'spectral_norm': [],
        'params': []
        # 'eigs':[],
        })
    if model_name and save_dir:
        save_path = os.path.join(save_dir, "_%s.csv"%(model_name.replace('/', '--')))
        if os.path.exists(save_path):
            layer_stats = pd.read_csv(save_path)
            for k in results.keys():
                results[k] = layer_stats[k].tolist()
    # print("=================================")
    # print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    # print("=================================")
    # iterate through layers
    module_names, module_shapes, modules = get_module_names_shapes(net, return_modules=True)
    # for name, m in net.named_modules():
    for name, m in zip(module_names, modules):
        if name in results["longname"]: continue
        # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type(m).__name__.lower() in ["conv2d", "conv1d", "linear"]:
            # if isinstance(m, nn.Linear) and max(m.weight.shape)/min(m.weight.shape) >= 8: continue # ignore classifier layer
            if type(m).__name__.lower() == "linear" and max(m.weight.shape)/min(m.weight.shape) >= 8: continue # ignore classifier layer
            matrix = m.weight.data.clone()
            matrix = matrix.cuda()
            # i have checked that the multiplication won't affect the weights value
            #print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            # if type(m).__name__.lower() in ["conv2d", "conv1d"]:
            if len(matrix.shape) > 2:
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            nz_eigs, final_D, final_alpha, alpha_weighted, entropy, log_alpha_norm, log_norm, log_spectral_norm, hard_rank, fnorm, spectral_norm, stable_rank = matrix_esd_estimator(matrix, m.weight.shape, filter_zeros, EVALS_THRESH, fix_fingers, xmin_pos, bins)
            del matrix

            results['D'].append(final_D)
            results['M'].append(m.weight.shape[0])
            results['N'].append(m.weight.shape[1])
            results['alpha'].append(final_alpha)
            results['alpha_weighted'].append(alpha_weighted)
            results['entropy'].append(entropy)
            results['log_alpha_norm'].append(log_alpha_norm)
            results['log_norm'].append(log_norm)
            results['log_spectral_norm'].append(log_spectral_norm)
            results['longname'].append(name)
            results['matrix_rank'].append(hard_rank)
            results['norm'].append(fnorm)
            results['num_evals'].append(len(nz_eigs))
            results['spectral_norm'].append(spectral_norm)
            results['stable_rank'].append(stable_rank)
            results['xmax'].append(nz_eigs[-1].item())
            results['xmin'].append(nz_eigs[0].item())
            # results['eigs'].append(eigs.detach().cpu().numpy())
            m_parameters = filter(lambda p: p.requires_grad, m.parameters())
            params = sum([np.prod(p.size()) for p in m_parameters])
            results['params'].append(params)

            if model_name and save_dir:
                layer_stats = pd.DataFrame({key:results[key] for key in results if key!='eigs'})
                save_path = os.path.join(save_dir, "_%s.csv"%(model_name.replace('/', '--')))
                layer_stats.to_csv(save_path)

    return results

# -------------------------
# Helpers
# -------------------------
def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def reset_peak_mem_all_devices():
    if not _cuda_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def get_peak_mem_all_devices():
    """
    Return peak allocated/reserved (bytes) as max over all devices.
    """
    if not _cuda_available():
        return 0, 0
    peak_alloc = 0
    peak_resv = 0
    for i in range(torch.cuda.device_count()):
        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated(i))
        peak_resv = max(peak_resv, torch.cuda.max_memory_reserved(i))
    return peak_alloc, peak_resv


def bytes_to_gb(x: int) -> float:
    return x / (1024**3)


def cleanup():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def build_model_args(model_name: str, extra: str = "") -> str:
    # NOTE: lm-eval splits by comma, so don't put commas inside values.
    base = [
        f"pretrained={model_name}",
        "trust_remote_code=True",
        "dtype=float16",
        "low_cpu_mem_usage=True",
        "device_map=auto",
        "parallelize=True",
    ]
    if extra:
        base.append(extra)
    return ",".join(base)


# -------------------------
# Method A: ESD (net_esd_estimator)
# -------------------------
def run_esd_method(model_name: str, esd_kwargs: dict):
    """
    Load model -> run net_esd_estimator -> report load_time, run_time, peak GPU mem.
    """
    if _cuda_available():
        reset_peak_mem_all_devices()

    # load
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if _cuda_available():
        torch.cuda.synchronize()
    load_time = time.perf_counter() - t0

    # measure from "load done" to "done"
    if _cuda_available():
        # If you want peak that includes model weights, DO NOT reset here.
        # If you want peak only during computation, reset here.
        # User asked "max GPU usage", usually include weights, so keep as-is.
        pass

    t1 = time.perf_counter()
    out = net_esd_estimator(model, **esd_kwargs)
    if _cuda_available():
        torch.cuda.synchronize()
    run_time = time.perf_counter() - t1

    peak_alloc, peak_resv = get_peak_mem_all_devices()

    # cleanup
    del model
    cleanup()

    return {
        "method": "net_esd_estimator",
        "load_time_sec": load_time,
        "run_time_sec": run_time,
        "peak_alloc_gb": bytes_to_gb(peak_alloc),
        "peak_reserved_gb": bytes_to_gb(peak_resv),
        "output": out if isinstance(out, (int, float, str, dict, list)) else str(type(out)),
    }


# -------------------------
# Method B: lm-eval (10 tasks)
# -------------------------
def run_lmeval_method(model_name: str, tasks: list[str], batch_size: int, model_args_extra: str):
    """
    Create HF model wrapper inside lm-eval (loads model) -> evaluate tasks -> report.
    Time window requested: load finished -> eval finished.
    We'll report both load_time and eval_time.
    """
    from lm_eval.models.huggingface import HFLM  # lm-eval internal wrapper

    tasks = [t.strip() for t in tasks if t.strip()]
    tasks = tasks[:10]

    if _cuda_available():
        reset_peak_mem_all_devices()

    # 1) load via HFLM so we can time load separately
    t0 = time.perf_counter()
    # tokenizer will be created internally by HFLM; if you need special tokenizer handling,
    # you can customize here.
    lm = HFLM(
        pretrained=model_name,
        trust_remote_code=True,
        dtype="float16",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if _cuda_available():
        torch.cuda.synchronize()
    load_time = time.perf_counter() - t0

    # 2) evaluate (time from "load done" to "eval done")
    t1 = time.perf_counter()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        model_args=None,  # since we passed an instantiated model wrapper
    )
    if _cuda_available():
        torch.cuda.synchronize()
    eval_time = time.perf_counter() - t1

    peak_alloc, peak_resv = get_peak_mem_all_devices()

    # cleanup
    del lm
    cleanup()

    return {
        "method": "lm_eval_10tasks",
        "tasks": tasks,
        "load_time_sec": load_time,
        "run_time_sec": eval_time,
        "peak_alloc_gb": bytes_to_gb(peak_alloc),
        "peak_reserved_gb": bytes_to_gb(peak_resv),
        "results": results,  # full lm-eval json
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)

    # outputs
    parser.add_argument("--output_dir", type=str, default="./results_compare")
    parser.add_argument("--batch_size", type=int, default=2)

    # lm-eval tasks (take first 10)
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,hellaswag,mmlu,truthfulqa_mc1,winogrande,piqa,openbookqa,boolq,lambada_openai,commonsense_qa",
    )

    # optional extra args placeholder (kept for compatibility)
    parser.add_argument("--model_args_extra", type=str, default="")

    # ESD parameters (pass-through)
    parser.add_argument("--esd_save_dir", type=str, default="./data/esd_out")
    parser.add_argument("--esd_fix_fingers", type=str, default="xmin_mid")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name
    safe_name = safe_model_name(model_name)

    out_path = os.path.join(args.output_dir, f"{safe_name}.compare.json")
    err_path = os.path.join(args.output_dir, f"{safe_name}.error.txt")

    print(f"[RUN] model={model_name}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"[INFO] cuda_available={_cuda_available()} device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # ESD kwargs (you can add more here later)
    esd_kwargs = dict(
        fix_fingers=args.esd_fix_fingers,
        save_dir=args.esd_save_dir,
        model_name=safe_name,
    )
    os.makedirs(args.esd_save_dir, exist_ok=True)

    all_out = {
        "model_name": model_name,
        "safe_name": safe_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "methods": [],
    }

    try:
        # Method A
        print("\n===== Method A: net_esd_estimator =====")
        esd_res = run_esd_method(model_name, esd_kwargs)
        all_out["methods"].append(esd_res)
        print(json.dumps({k: v for k, v in esd_res.items() if k != "output"}, indent=2))

        # Method B
        print("\n===== Method B: lm_eval (first 10 tasks) =====")
        lmeval_res = run_lmeval_method(
            model_name=model_name,
            tasks=tasks,
            batch_size=args.batch_size,
            model_args_extra=args.model_args_extra,
        )
        all_out["methods"].append({
            **{k: v for k, v in lmeval_res.items() if k != "results"},
            "summary_table": make_table(lmeval_res["results"]),
        })
        print(make_table(lmeval_res["results"]))

        # Save full results (including lm-eval raw json)
        all_out["lm_eval_raw"] = lmeval_res["results"]

        with open(out_path, "w") as f:
            json.dump(all_out, f, indent=2)

        if os.path.exists(err_path):
            os.remove(err_path)

        print(f"\n[SAVED] {out_path}")

    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR]\n{tb}")
        with open(err_path, "w") as f:
            f.write(tb)
        raise

    finally:
        cleanup()


if __name__ == "__main__":
    main()
