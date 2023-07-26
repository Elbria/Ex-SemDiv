import argparse
import numpy as np
import torch
import json
import random
from tqdm import tqdm, trange
import pickle
import copy
import csv
from IPython.core.display import display, HTML

from lime.lime_text import LimeTextExplainer
from transformers import glue_processors as processors
from transformers import semdiv_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassificationMarginLoss,
                                  BertForSequenceClassification, BertTokenizer)

MODEL_CLASSES = {
    'bert_margin': (BertConfig, BertForSequenceClassificationMarginLoss, BertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def colorize_twoway(words, color_array, max_width_shown=600):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    template_pos = '<span class="barcode"; style="color: black; background-color: rgba(255, 0, 0, {}); display:inline-block;">{}</span>'
    template_neg = '<span class="barcode"; style="color: black; background-color: rgba(0, 0, 255, {}); display:inline-block;">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        if color > 0:

            colored_string += template_neg.format(-color, '&nbsp' + word + '&nbsp')
    return '<div style="width:%dpx">' % max_width_shown + colored_string + '</div>'

def format_explanations(explanations):
    explanations = explanations.as_map()[1]
    ordered_explanations = np.zeros(len(explanations))
    for idx, v in explanations:
        ordered_explanations[idx] = v * -1.
    return ordered_explanations

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def explain_instance(args, model, tokenizer, explainer, text_a, text_b, explain_source):
    def predict_proba(texts, text_a=text_a, text_b=text_b):
        if explain_source:
            texts_a, texts_b = texts, [text_b]*len(texts)
        else:
            texts_a, texts_b = [text_a]*len(texts), texts
        features = []
        pad_token, pad_token_segment_id = 0, 0
        for (text_a, text_b) in zip(texts_a, texts_b):
            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=args.max_length,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1] * len(input_ids)
            # Zero-pad up to first sequence length.
            padding_length = args.max_length - len(input_ids)

            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == args.max_length, "Error with input length {} vs {}".format(len(input_ids), args.max_length)
            assert len(attention_mask) == args.max_length, "Error with input length {} vs {}".format(len(attention_mask), args.max_length)
            assert len(token_type_ids) == args.max_length, "Error with input length {} vs {}".format(len(token_type_ids), args.max_length)
            features.append(InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=1))
        all_input_ids_a = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask_a = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids_a = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_a = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, all_labels_a)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        preds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids_a': batch[0],
                          'attention_mask_a': batch[1],
                          'labels_a': batch[3],
                          'input_ids_b': None,
                          'attention_mask_b': None,
                          'labels_b': None,
                          'token_type_ids_a': None,
                          'token_type_ids_b': None}
                logits_a = model(**inputs)

                if preds is None:
                    preds = logits_a.detach().cpu().numpy()
                    out_label_ids = inputs['labels_a'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits_a.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels_a'].detach().cpu().numpy(), axis=0)

        sigm = [1 / (1 + np.exp(-x)) for x in preds]
        equiv, div = [], []
        for id_, x in enumerate(sigm):
            equiv.append(x)
            div.append(1-x)
        proba = np.column_stack((equiv, div))
        return proba

    if explain_source:
        explanations = explainer.explain_instance(text_a, predict_proba, num_features=len(text_a.split()), num_samples=args.num_samples, labels=(1, ))
    else:
        explanations = explainer.explain_instance(text_b, predict_proba, num_features=len(text_b.split()), num_samples=args.num_samples, labels=(1, ))
    explanations = format_explanations(explanations)
    return explanations

def explain_sentence(args, model, tokenizer, text_a, text_b, explain_source):
    explainer = LimeTextExplainer(class_names=['0', '1'], bow=False, split_expression=' ', mask_string='[MASK]', verbose=False)
    explanations = explain_instance(args, model, tokenizer, explainer, text_a, text_b, explain_source)
    return explanations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg")
    parser.add_argument("--tokenizer_name", default="bert-base-multilingual-cased")
    parser.add_argument("--model_type", default="bert_margin")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--task_name", default="SemDiv")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_length", default=128)
    parser.add_argument("--num_labels", default=2)
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_samples", default=1000)
    parser.add_argument("--bitexts", default="/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/annotations/sd_inter_pos_both")
    parser.add_argument("--output", default="/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/outputs/lime")
    #parser.add_argument("--bitexts", default="/fs/clip-divergences/xling-SemDiv/for_divergentmBERT/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_multi_hard/rdpg/test_synthetic.div.tsv")
    #parser.add_argument("--output", default="/fs/clip-divergences/xling-SemDiv/for_divergentmBERT/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_multi_hard/rdpg/test_synthetic.div.lime")
    parser.add_argument("--synthetic", default=True)
    parser.add_argument('--margin', type=float, default=0.2, help="margin for triplet loss")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Load tokenizer and model to explain
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels, finetuning_task=args.task_name)
    config.margin = args.margin
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model.to(args.device)
    texts_a_id, texts_b_id, golds_a_id, golds_b_id = 0, 1, 2, 3
    #if args.synthetic: texts_a_id, texts_b_id, golds_a_id, golds_b_id = 1, 2, 3, 4
    texts_a, texts_b, golds_a, golds_b, explanations_a, explanations_b, golds_a_, golds_b_ = [], [], [], [], [], [], [], []
    with open(args.bitexts, 'r') as file:
        lines = file.readlines()
        #tsv_file = csv.reader(file, delimiter="\t")
        for id_, line in enumerate(lines):
            line = line.split('\t')
            #if id_ > 0:
            #    continue
            texts_a.append(line[texts_a_id])
            texts_b.append(line[texts_b_id])
            golds_a.append(line[golds_a_id])
            golds_b.append(line[golds_b_id])


    for id_,(text_a, text_b) in enumerate(zip(texts_a, texts_b)):
        print('Pair number: ' + str(id_))
        expl_source = explain_sentence(args, model, tokenizer, text_a, text_b, explain_source=True)
        expl_target = explain_sentence(args, model, tokenizer, text_a, text_b, explain_source=False)

        explanations_adjusted_source = np.array(expl_source) / max(np.abs(np.array(expl_source)))
        explanations_a.append(list(explanations_adjusted_source))
        explanations_adjusted_target = np.array(expl_target) / max(np.abs(np.array(expl_target)))
        explanations_b.append(list(explanations_adjusted_target))
        gold_a = [x for x in golds_a[id_].split(' ')]
        gold_b = [x for x in golds_b[id_].split(' ')]
        golds_a_.append(gold_a)
        golds_b_.append(gold_b)



    print(f'Writing results to {args.output}')
    with open(args.output, 'w') as output_f:
        for id_ in range(len(explanations_a)):
            line = lines[id_].split('\t')
            row = []
            for l in line:
                row.append(l.rstrip())
            row.append(' '.join([str(round(a, 3)) for a in explanations_a[id_]]))
            row.append(' '.join([str(round(a, 3)) for a in explanations_b[id_]]))
            row_ = '\t'.join(row)
            output_f.write(f'{row_}\n')
    output_f.close()

if __name__ == "__main__":
    main()
