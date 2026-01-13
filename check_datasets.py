import os
import sys
import dataclasses
from dataclasses import dataclass

# 1. Force environment to trust legacy scripts
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

print(f"--- Environment Check (Python {sys.version.split()[0]}) ---")

# TEST A: Dataclass Strictness
# This checks if Python 3.12 can inspect a class properly
try:
    @dataclass
    class TestData:
        id: int
        label: str

    fields = dataclasses.fields(TestData)
    print("✅ TEST A: Dataclasses are working correctly.")
except TypeError as e:
    print(f"❌ TEST A FAILED: Dataclass inspection error: {e}")

# TEST B: Dataset Loading (Social IQa)
# This checks if 'datasets' can handle the legacy .py scripts
try:
    from datasets import load_dataset
    print("Attempting to load Social IQa configuration...")
    # We only load the metadata/info to save time and bandwidth
    ds_builder = load_dataset("social_i_qa", trust_remote_code=True)
    print("✅ TEST B: Social IQa script loaded successfully.")
except Exception as e:
    print(f"❌ TEST B FAILED: Could not load legacy dataset script: {e}")

# TEST C: Triton & Torch Alignment
try:
    import torch
    import triton
    print(f"Torch Version: {torch.__version__}")
    print(f"Triton Version: {triton.__version__}")
    if triton.__version__ == "3.1.0":
        print("✅ TEST C: Triton 3.1.0 detected (Recommended for Python 3.12).")
    else:
        print(f"⚠️ TEST C WARNING: Triton is {triton.__version__}. If you see Dataclass errors, downgrade to 3.1.0.")
except ImportError as e:
    print(f"⚠️ TEST C: Could not verify Torch/Triton: {e}")

print("\n--- Check Complete ---")