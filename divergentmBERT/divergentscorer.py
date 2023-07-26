import torch
from tqdm import tqdm
import numpy as np

import copy
import json

from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import semdiv_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassificationMarginLoss,
                                  BertForSequenceClassification, BertTokenizer)
MODEL_CLASSES = {
    'bert_margin': (BertConfig, BertForSequenceClassificationMarginLoss, BertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
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

class DivergentmBERTScorer:

    def __init__(
        self,
        model_path,
        tokenizer_path,
        do_lower_case,
        device,
        model_type='bert_margin',
        task_name='SemDiv',
        num_labels=2,
        margin=0.5,
        max_length=128,
        eval_batch_size=64
    ):
        self.num_labels = num_labels
        self.margin = margin
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.task_name = task_name

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_path, num_labels=num_labels, finetuning_task=task_name)
        config.margin = margin
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
        self.model = model_class.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=config)
        self.model.to(self.device)

    def compute_divergentscore(self, texts_a, texts_b):

        processor = processors[self.task_name.lower()]()
        examples = []

        for (i, (text_a, text_b)) in enumerate(zip(texts_a, texts_b)):

            guid = "%s-%s" % ('test', i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label='0'))

        features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=processor.get_labels(),
                                                max_length=self.max_length,
                                                output_mode=output_modes[self.task_name.lower()],
                                                pad_on_left=False,
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                test=True
                                                )

        all_input_ids_a = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask_a = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids_a = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_a = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, all_labels_a)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        preds = None
        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids_a': batch[0],
                          'attention_mask_a': batch[1],
                          'labels_a': batch[3],
                          'input_ids_b': None,
                          'attention_mask_b': None,
                          'labels_b': None,
                          'token_type_ids_a': None,
                          'token_type_ids_b': None}

                logits_a = self.model(**inputs)
                if preds is None:
                    preds = logits_a.detach().cpu().numpy()
                    out_label_ids = inputs['labels_a'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits_a.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels_a'].detach().cpu().numpy(), axis=0)
        return preds

    def compute_divergentscore_(self, texts_a, texts_b):

        processor = processors[self.task_name.lower()]()

        examples = []
        for (i, (text_a, text_b)) in enumerate(zip(texts_a, texts_b)):

            guid = "%s-%s" % ('test', i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label='0'))


        features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=processor.get_labels(),
                                                max_length=self.max_length,
                                                output_mode=output_modes[self.task_name.lower()],
                                                pad_on_left=False,
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                test=True
                                                )

        all_input_ids_a = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask_a = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids_a = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_a = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, all_labels_a)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        preds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids_a': batch[0],
                          'attention_mask_a': batch[1],
                          'labels_a': batch[3],
                          'input_ids_b': None,
                          'attention_mask_b': None,
                          'labels_b': None,
                          'token_type_ids_a': None,
                          'token_type_ids_b': None}

                logits_a = self.model(**inputs)
                if preds is None:
                    preds = logits_a.detach().cpu().numpy()
                    out_label_ids = inputs['labels_a'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits_a.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels_a'].detach().cpu().numpy(), axis=0)

        return preds