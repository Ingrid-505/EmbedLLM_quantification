import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 1. PATH CONFIGURATION (Matches repo defaults)
DATA_DIR = "./data" # Or your preferred data folder
os.makedirs(DATA_DIR, exist_ok=True)

# Load your raw correctness matrix
# (Rows = Models, Columns = Questions)
full_df = pd.read_csv("train_correctness.csv", index_col=0)

# 2. CREATE ORDER FILES (Critical for index-matching)
# model_order.csv must be a single column of model names
model_names = full_df.index.tolist()
pd.DataFrame(model_names).to_csv(os.path.join(DATA_DIR, "model_order.csv"), index=False, header=False)

# 3. SPLIT QUESTIONS (Vertical Split)
all_questions = full_df.columns.tolist()
train_qs, test_qs = train_test_split(all_questions, test_size=0.20, random_state=42)

# question_order.csv must be the CONCATENATED list of [Train, Test]
# This allows the 'get_question_embedding_tensor.py' to map them to the matrix
ordered_questions = train_qs + test_qs
pd.DataFrame(ordered_questions).to_csv(os.path.join(DATA_DIR, "question_order.csv"), index=False, header=False)

# 4. EXPORT MATRICES
# The repo algorithm reads these as the ground truth
full_df[train_qs].to_csv(os.path.join(DATA_DIR, "train.csv"))
full_df[test_qs].to_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"✅ Data prepared and saved to {DATA_DIR}")
print(f"Next: Run the embedding script on {len(ordered_questions)} questions.")