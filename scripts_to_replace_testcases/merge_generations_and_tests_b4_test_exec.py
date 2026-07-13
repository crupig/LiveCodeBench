import os
import re
import json
import pandas as pd
import gzip

if __name__ == "__main__":
    
    # IMPORT CODE KNOWLEDGE BASE
    kb_dir = "../data/knowlbase/LiveCodeBench"
    kb = pd.DataFrame()
    for file in os.listdir(kb_dir):
        if re.search(r'knowlbase_livecodebench_part\d+\.json', file):
            print(f"Processing file: {file}")
            knowlbase_path = os.path.join(kb_dir, file)
            with open(knowlbase_path, "r") as f:
                kb_j = json.load(f)
            kb_t = pd.DataFrame(kb_j)
            kb = pd.concat([kb, kb_t], ignore_index=True)

    kb = kb[['solution_idx', 'task_idx', 'generated_by', 'prompt', 'method']]
    kb['onwhichtomerge'] = kb['solution_idx'].apply(lambda x: x.split("--SampleID")[0])
    
    # FILTER KNOWLEDGE BASE TO TEST SET SOLUTIONS
    with open("../data/testset/test.jsonl", "r") as f:
        test_set = [json.loads(line) for line in f]
    test_set = pd.DataFrame(test_set)
    solution_ids_to_keep = test_set.solution_idx.unique().tolist()
    kb = kb[kb.solution_idx.isin(solution_ids_to_keep)]
    del test_set
    del solution_ids_to_keep

    
    # SPLIT BY GENERATED_BY
    for generated_by in kb['generated_by'].unique():
        kb_genby = kb[kb['generated_by'] == generated_by]

        # IMPORT TESTS KNOWLEDGE BASE
        test_path = f"../data/knowlbase-tests/{generated_by}_knowlbase_tests_livecodebench.jsonl"
        
        if not os.path.exists(test_path):
            print(f"Test knowledge base file not found for {generated_by} at path: {test_path}. Skipping.")
            continue

        with open(test_path, "r") as f:
            test_j = [json.loads(line) for line in f]
        kbt = pd.DataFrame(test_j)

        # kbt = pd.read_json(test_path, lines=True).drop(columns=['prompt'])
        kbt['onwhichtomerge'] = kbt['test_idx'].apply(lambda x: x.split("--TestID")[0])
        kbt = kbt.drop(columns=[
            'task_idx', 
            'generated_by', 
            # 'num_unique_asserts',
        ])

        # merging with code generations
        merged = kb_genby.merge(kbt, on="onwhichtomerge", how="inner")

        ###############
        num_task_ids = json.load(open('../../../constants/ids_train_val_test.json', 'r'))['LiveCodeBench']['test']
        g = merged.groupby('solution_idx', as_index=False).agg({'test_idx': 'count', 'task_idx': 'first'})
        print(f"################## {generated_by} ##################")
        print(kb_genby.loc[kb_genby['task_idx'].isin(num_task_ids)]['task_idx'].nunique())
        print(f"Max number of unique asserts per task ID:\t\t{g['test_idx'].max()}")
        print(f"Min number of unique asserts per task ID:\t\t{g['test_idx'].min()}")
        print(f"Average number of unique asserts per task ID:\t\t{g['test_idx'].mean():.2f}")
        print(f"Median number of unique asserts per task ID:\t\t{g['test_idx'].median()}")
        print(f"Number of task IDs with no asserts (out of {len(num_task_ids)}):\t{len(num_task_ids) - g['task_idx'].nunique()}")
        print()
        ###############
        
        merged['test_execution_idx'] = merged.apply(lambda row : \
        '{}--TestID::{}--SampleID::{}'.format(
                row['solution_idx'].split('--SampleID::')[0],
                row['test_idx'].split('--TestID::')[-1],
                row['solution_idx'].split('--SampleID::')[-1]
            ), axis=1
        )

        g = merged.groupby('test_idx', as_index=False).agg({
            'test_input' : 'first',
            'test_output' : 'first',
            'task_idx' : 'first',
            'test_execution_idx' : list,
            'solution_idx' : list,
            'method' : list,
            'test_statement' : 'first',
            'generated_by' : 'first',
        }).rename(columns={'method': 'code_list'})

        
        # authorize saving form input
        save_dir = "../test-4-execution"
        authorized = input(f"Do you want to save the tests extracted for {generated_by} to '{save_dir}'? (y/n): ")
        if authorized.lower() in ['y', 'yes']:
        
            # chunk merged and save to json
            chunk_size = 200
            num_chunks = (len(g) + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk = g.iloc[i*chunk_size:(i+1)*chunk_size]
                chunk.to_json(os.path.join(save_dir, f"{generated_by}_test4execution_{i:02d}.json"), orient="records", indent=2)
            
            print(f"Saved merged data to {save_dir} in {num_chunks} chunks...")
        else:
            print("Skipped.")