# # # import pandas as pd

# # # # Load the list we made earlier
# # # df = pd.read_csv("data/question_order.csv", header=None)

# # # # Name the column 'prompt' so the repo script can find it
# # # df.columns = ['prompt']

# # # # Save it back WITH the header
# # # df.to_csv("data/question_order.csv", index=False)
# # # print("✅ question_order.csv updated with 'prompt' header.")
# # import pandas as pd
# # import os

# # def convert_to_long_format(input_path, output_path):
# #     # Load your matrix (Rows=Models, Cols=Questions)
# #     df = pd.read_csv(input_path, index_col=0)
    
# #     # We need numeric IDs for the model_id column
# #     # Create a mapping based on your model_order.csv
# #     model_order = pd.read_csv("data/model_order.csv", header=None)[0].tolist()
# #     model_to_id = {name: i for i, name in enumerate(model_order)}
    
# #     # Create a mapping for questions based on question_order.csv
# #     question_order = pd.read_csv("data/question_order.csv", header=0)['prompt'].tolist()
# #     question_to_id = {q: i for i, q in enumerate(question_order)}

# #     long_data = []
# #     for model_name, row in df.iterrows():
# #         m_id = model_to_id[model_name]
# #         for q_text, score in row.items():
# #             if q_text in question_to_id:
# #                 q_id = question_to_id[q_text]
# #                 long_data.append({
# #                     "model_id": m_id,
# #                     "question_id": q_id,
# #                     "correctness": score
# #                 })
    
# #     pd.DataFrame(long_data).to_csv(output_path, index=False)
# #     print(f"✅ Converted {input_path} to long format at {output_path}")

# # # Run the conversion
# # convert_to_long_format("data/train.csv", "data/train_long.csv")
# # convert_to_long_format("data/test.csv", "data/test_long.csv")
# import pandas as pd
# import os

# def convert_to_final_long_format(input_path, output_path):
#     # Load your wide matrix
#     df = pd.read_csv(input_path, index_col=0)
    
#     # Load orders to ensure ID consistency
#     model_order = pd.read_csv("data/model_order.csv", header=None)[0].tolist()
#     model_to_id = {name: i for i, name in enumerate(model_order)}
    
#     question_order = pd.read_csv("data/question_order.csv", header=0)['prompt'].tolist()
#     question_to_id = {q: i for i, q in enumerate(question_order)}

#     long_data = []
#     for model_name, row in df.iterrows():
#         m_id = model_to_id[model_name]
#         for q_text, score in row.items():
#             if q_text in question_to_id:
#                 q_id = question_to_id[q_text]
#                 long_data.append({
#                     "model_id": m_id,
#                     "model_name": model_name,  # The missing column
#                     "question_id": q_id,
#                     "correctness": score
#                 })
    
#     pd.DataFrame(long_data).to_csv(output_path, index=False)
#     print(f"✅ Re-formatted {input_path} with 'model_name' at {output_path}")

# # Execute conversion
# convert_to_final_long_format("data/train.csv", "data/train_long.csv")
# convert_to_final_long_format("data/test.csv", "data/test_long.csv")
import pandas as pd
import os

def convert_to_repo_schema(input_path, output_path):
    # Load your matrix (Rows=Models, Cols=Questions)
    df = pd.read_csv(input_path, index_col=0)
    
    # Load orders to ensure IDs align with your embeddings
    model_order = pd.read_csv("data/model_order.csv", header=None)[0].tolist()
    model_to_id = {name: i for i, name in enumerate(model_order)}
    
    # Load questions (Assumes we fixed the 'prompt' header in the previous turn)
    question_order = pd.read_csv("data/question_order.csv")['prompt'].tolist()
    question_to_id = {q: i for i, q in enumerate(question_order)}

    long_data = []
    for model_name, row in df.iterrows():
        if model_name not in model_to_id:
            continue
        m_id = model_to_id[model_name]
        
        for q_text, score in row.items():
            if q_text in question_to_id:
                p_id = question_to_id[q_text]
                long_data.append({
                    "model_id": m_id,
                    "model_name": model_name,
                    "prompt_id": p_id, # Changed from question_id to prompt_id
                    "correctness": int(score)
                })
    
    # Save with the specific schema the script looks for
    pd.DataFrame(long_data).to_csv(output_path, index=False)
    print(f"✅ Created {output_path} with 'prompt_id' and 'model_name' columns.")

# Execute conversion
convert_to_repo_schema("data/train.csv", "data/train_long.csv")
convert_to_repo_schema("data/test.csv", "data/test_long.csv")