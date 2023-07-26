import sys

sys.path.append('./divergentmBERT')
import pandas as pd
import numpy as np
import shap
import os
import argparse
from scorer import DivergentmBERTScorer

import torch

class DivergentmBERTWrapper():
    def __init__(self, model_path, tokenizer_path, do_lower_case, device, explain_source):
        self.scorer = DivergentmBERTScorer(model_path, tokenizer_path, do_lower_case, device)
        self.explain_source = explain_source
        self.text_a = None
        self.text_b = None

    def __call__(self, translations):
        if self.explain_source:
           
            target = [self.text_b] * len(translations)
            divergent_scores = self.scorer.compute_divergentscore(translations, target, self.explain_source)
        else:
            source = [self.text_a] * len(translations)
            divergent_scores = self.scorer.compute_divergentscore(source, translations, self.explain_source)
        return np.array(divergent_scores)

    def tokenize_sent(self, sentence):
        return sentence.split(' ')

    def detokenize_sent(self, tokens):
        return ' '.join(tokens)

    def build_feature(self, trans_sent):
        tokens = self.tokenize_sent(trans_sent)
        tdict = {}
        for i, tt in enumerate(tokens):
            tdict['{}_{}'.format(tt, i)] = tt

        df = pd.DataFrame(tdict, index=[0])
        return df

    def mask_model(self, mask, x):
        tokens = []
        #print(f'MASK: {mask}')
        #print(f'{x}\n')
        #exit(0)
        for mm, tt in zip(mask, x):
            if mm:
                tokens.append(tt)
            else:
                #continue
                tokens.append('[MASK]')
        trans_sent = self.detokenize_sent(tokens)
        sentence = pd.DataFrame([trans_sent])
        #print(f'Sentence: {trans_sent}\n')
        return sentence

class ExplainableDivergentmBERT():
    def __init__(self, model, tokenizer, do_lower_case, device,  explain_source):
        self.wrapper = DivergentmBERTWrapper(model, tokenizer, do_lower_case, device, explain_source)
        self.explainer = shap.Explainer(self.wrapper, self.wrapper.mask_model)
        self.explain_source = explain_source

    def __call__(self, text_a, text_b):
        if self.explain_source:
            return self.wrapper(text_a)
        else:
            return self.wrapper(text_b)

    def explain(self, text_a, text_b, plot=False):
        if self.explain_source: text = text_a
        else: text = text_b
        value = self.explainer(self.wrapper.build_feature(text))
        if plot: shap.waterfall_plot(value[0])
        all_tokens = self.wrapper.tokenize_sent(text)

        return [[token, sv] for token, sv in zip(all_tokens, value[0].values)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg")
    parser.add_argument("--input", default="/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/annotations/sd_inter_pos_both")
    parser.add_argument("--output", default="/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/outputs/shap")
    parser.add_argument("--tokenizer_name", default="bert-base-multilingual-cased")
    parser.add_argument("--model_type", default="bert_margin")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--task_name", default="SemDiv")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    texts_a, texts_b = [], []
    with open(args.input, encoding='utf-8') as input_f:
        lines = input_f.readlines()
        verbose = False
        for id_, line in enumerate(lines):
            #if id_ > 0:
            #   continue
            line = line.split('\t')
            texts_a.append(line[0].rstrip())
            texts_b.append(line[1].rstrip())

    explanations_source, explanations_target = [], []
    count =0
    for explain_source in [True, False]:
        # Initialize explainable DivergentmBERT model
        print('Initialize Explainable DivergentmBERT')
        model = ExplainableDivergentmBERT(model=args.model_name_or_path, tokenizer=args.tokenizer_name,
                                                 do_lower_case=args.do_lower_case, device=args.device,
                                                 explain_source=explain_source)
        print('Instance initialized...')
        for (text_a, text_b) in zip(texts_a, texts_b):
            count+=1
            print(count)
            exp_scores =  []
            model.wrapper.text_a = text_a
            model.wrapper.text_b = text_b

            exps = model.explain(text_a, text_b)
            if explain_source:
                explanations_source.append([float(entry[1]) for entry in exps])
            else:
                explanations_target.append([float(entry[1]) for entry in exps])


    print(f'Writing results to {args.output}')
    with open(args.output, 'w') as output_f:
        for id_ in range(len(explanations_source)):
            line = lines[id_].split('\t')
            row = []
            for l in line:
                row.append(l.rstrip())
            row.append(' '.join([str(round(a, 3)) for a in explanations_source[id_]]))
            row.append(' '.join([str(round(a, 3)) for a in explanations_target[id_]])) 
            row_ = '\t'.join(row)
            output_f.write(f'{row_}\n')
    output_f.close()

if __name__ == "__main__":
    main()


