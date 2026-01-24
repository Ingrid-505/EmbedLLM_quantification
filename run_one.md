## Environment
`conda create -n embedllm-eval python=3.10 -y`
`conda activate embedllm-eval`
`pip install -r requirements_min.txt`

## HF settings
`export HF_TOKEN=xxxx`
`export HF_HOME=/path/to/cache/hf`
`export HF_DATASETS_CACHE=$HF_HOME/datasets`
`export TRANSFORMERS_CACHE=$HF_HOME/transformers`

## Sample

`CUDA_VISIBLE_DEVICES=0 python run_one.py --model_name <MODEL_NAME> --batch_size 2`
or run with run_evals.sh after changing `CSV_PATH` and `GPU_GROUPS`
`bash run_evals.sh`