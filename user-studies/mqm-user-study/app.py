from flask import Flask, request
from flask import render_template
#from flask_ngrok import run_with_ngrok
import time
import json
import random

# Adapted from Navita
# TODO: Change response directory for Spanish
# TODO: Change completion code
# TODO: Change language in startup.js
# TODO: Change language in tutorial index.html

config = {
    1: {'w_hl': True},
    2: {'w_hl': False},
}

tracker_dir = 'tracker/'

samples_tracked = {
    1: json.load(open('tracker/v1.json', 'r')),
    2: json.load(open('tracker/v2.json', 'r')),
}

num_samples= 25
num_responses = 1

def sample(examples, num_samples):
    examples = {key:value for key, value in examples.items() if value>-1}
    values = [value for key, value in examples.items() if value<num_responses]
    if values:
        candidates = [key for key,value in examples.items() if value==min(values)]
    else:
        return None
    if len(candidates) < num_samples:
        return None
    samples = random.sample(candidates, num_samples)
    print(samples)
    samples = [int(e) for e in samples]
    return samples

def condition_to_idx(data):
    w_hl = data['with_hl']=='true'
    if w_hl:
        return 1
    else:
        return 2


def idx_to_condition(idx):
    w_hl = 'true' if config[idx]['w_hl'] else 'false'
    return w_hl

def save_response(data):
    save_name = time.strftime("%Y%m%d-%H%M%S")
    response_file = f'responses/ced_gpt/gpt_lit_en_pt_ced_0/{save_name}.json'
    with open(response_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    samples = data['sample_indices'].split(",")

    condition = condition_to_idx(data)

    for example in samples:
        if example in samples_tracked[condition]:
            samples_tracked[condition][example] = abs(samples_tracked[condition][example])
        else:
            print('Error')

    with open(f'{tracker_dir}/v{condition}.json', 'w') as f:
        json.dump(samples_tracked[condition], f, ensure_ascii=False, indent=4)
    return 'C103QV6Q'

def hold_examples(tracker, examples):
    for ins in examples:
        if tracker[f'{ins}'] == 0:
            tracker[f'{ins}'] = -1
        else:
            tracker[f'{ins}'] = -abs(tracker[f'{ins}'] + 1)

def get_examples():
    candidates = {key:len([i for i in value if value[i]==0]) for key, value in samples_tracked.items()}
    # First round of annotations in!
    if candidates[1] == 0 and candidates[2] == 0:
        candidates = {key: len([i for i in value if value[i] == 1]) for key, value in samples_tracked.items()}
        # Second round of annotations in!
        if candidates[1] == 0 and candidates[2] == 0:
            candidates = {key: len([i for i in value if value[i] == 2]) for key, value in samples_tracked.items()}
    condition = max(candidates, key=candidates.get)
    print(candidates)
    print(condition)

    examples = sample(samples_tracked[condition], num_samples)
    if examples is None:
        return None, None
    hold_examples(samples_tracked[condition], examples)

    with open(f'{tracker_dir}/v{condition}.json', 'w') as f:
        json.dump(samples_tracked[condition], f, ensure_ascii=False, indent=4)

    w_hl = idx_to_condition(condition)
    
    return examples, w_hl

# ----------------------------------------------------------- #
app = Flask(__name__, static_url_path='/static')
#run_with_ngrok(app)
app.templates_auto_reload = True
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=='POST':
        form = request.form
        data = form.to_dict()

        save_name = save_response(data)
        return save_name
        
    else:
        examples, w_hl = get_examples()
        if examples is None:
            return render_template('error.html')
        random.shuffle(examples)
        return render_template('index.html', params={'examples':examples, 'w_hl':w_hl})


if __name__ == "__main__":
    print({key: len([i for i in value if value[i] == 0]) for key, value in samples_tracked.items()})
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=4001)

