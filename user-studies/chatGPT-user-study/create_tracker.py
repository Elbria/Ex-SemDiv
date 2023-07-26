import json

total_examples = 50
num_conditions = 2

for idx in range(1, num_conditions+1):
    examples = {i:0 for i in range(1, total_examples+1)}
    json.dump(examples, open(f'tracker/v{idx}.json', 'w'), ensure_ascii=False, indent=4)
