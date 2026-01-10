import pandas as pd

# 1. Load your uploaded file
df = pd.read_csv('truthfulqa.csv')

# 2. Extract only the Question part (removing Answer and the Q: prefix)
def extract_prompt(statement):
    # Split by the tab separator for the Answer
    if '\tA:' in statement:
        q_part = statement.split('\tA:')[0]
    else:
        q_part = statement
    
    # Remove "Q: " prefix
    if q_part.startswith('Q: '):
        q_part = q_part[3:]
    elif q_part.startswith('Q:'):
        q_part = q_part[2:]
    
    return q_part.strip()

df['prompt'] = df['statement'].apply(extract_prompt)

# 3. Format exactly for the EmbedLLM repo (prompt_id, prompt)
question_order = df[['prompt']].copy()
question_order.index.name = 'prompt_id'
question_order = question_order.reset_index()

# 4. Save the file
question_order.to_csv('question_order.csv', index=False)
print("Successfully created question_order.csv with 3,178 prompts.")