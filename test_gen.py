from datasets import load_dataset
import pandas as pd
import json

dataset = load_dataset("vdey/MuMOInstruct")

task = 'bbbp+drd2+qed'
setting = 'seen'
test_data = load_dataset("vdey/MuMOInstruct")['test']
test_data = pd.DataFrame(test_data)
test_data = test_data[(test_data['task'] == task) & (test_data['instr_setting'] == setting)]
data=[]
for row in test_data.itertuples():
    item = {
        "source_smiles": row.source_smiles,
        "task": row.task,
        "instr_setting": row.instr_setting,
        "target_smiles": row.target_smiles,
        "properties": row.properties,
        "instr_idx": row.instr_idx,
        "split": row.split
    }
    data.append(item)

json.dump(data, open(f"test.json", "w"), indent=2)