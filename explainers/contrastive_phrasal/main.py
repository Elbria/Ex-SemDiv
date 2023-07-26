import sys
sys.path.append('/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/divergentmBERT')
from nltk.translate.phrase_based import phrase_extraction
import argparse
from divergentscorer import DivergentmBERTScorer
import torch
import utils
import numpy as np

class DivergentmBERT():
    def __init__(self, model, tokenizer, do_lower_case, device):
        self.scorer = DivergentmBERTScorer(model, tokenizer, do_lower_case, device)

    def __call__(self, text_a, text_b):
        return self.scorer.compute_divergentscore(text_a, text_b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default="/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg")
    parser.add_argument("--tokenizer_name", default="bert-base-multilingual-cased")
    parser.add_argument("--model_type", default="bert_margin")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--task_name", default="SemDiv")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_length", default=128)
    parser.add_argument("--num_labels", default=2)
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_samples", default=1000)
    parser.add_argument("--input", default="/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/annotations/nd_inter_pos_both")
    parser.add_argument("--output",
                        default="/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/outputs/LeavePhrasePairOut_inter_2delete")
    parser.add_argument("--synthetic", default=True)
    parser.add_argument('--margin', type=float, default=0.2, help="margin for triplet loss")
    parser.add_argument("--missing_phrases", action='store_true')
    parser.add_argument("--paired_phrases", action='store_true')
    parser.add_argument("--reranking", action='store_true')
    parser.add_argument("--filtering", action='store_true')
    parser.add_argument("--parse", action='store_true')
    parser.add_argument("--filtering_ngrams", action='store_true')
    parser.add_argument("--mask_token", action='store_true')
    parser.add_argument("--pos", action='store_true')
    parser.add_argument("--reward", action='store_true')
    parser.add_argument("--max_ngram", type=int, default=5)
    parser.add_argument("--reverse", default=False)
    parser.add_argument("--kernel_width", type=int, default=25)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not True else "cpu")
    device = "cuda"

    model = DivergentmBERT(model=args.model_name_or_path,
                           tokenizer=args.tokenizer_name,
                           do_lower_case=False,
                           device=device)

    print(f'Missing phrases: {args.missing_phrases}')
    print(f'Paired phrases: {args.paired_phrases}')
    print(f'Reranking: {args.reranking}')
    print(f'Pos: {args.pos}')
    reverse=args.reverse
    divs_lpo = []
    with open(args.input, encoding='utf-8') as input_f:
        lines = input_f.readlines()
        verbose = False
        for id_, line in enumerate(lines):
            line = line.split('\t')
            print(f'____________________________________{id_}___________________________________________________')
            if not reverse:
                en_text = line[0]
                fr_text = line[1]
                en_text_original, fr_text_original = en_text, fr_text
                #en_div = line[2]
                #fr_div = line[3]
                en_div = en_text
                fr_div = fr_text
            else:
                en_text = line[1]
                fr_text = line[0]
                en_div = line[3]
                fr_div = line[2]
                en_text_original, fr_text_original = en_text, fr_text
            en_len = len(en_text.split(' '))*0.5
            fr_len = len(fr_text.split(' '))*0.66
            alis = line[2] #4
            if args.parse:
               syntactic_phrases = utils.extract_syntactic_phrases(en_text, line[5].rstrip())
            if args.pos:
               en_excluded, fr_excluded = utils.extract_invalid_pos(en_text, line[-2].rstrip(), fr_text, line[-1].rstrip())
            en2fr = utils.ali_to_tuple(alis)
            phrases = set()
            if args.paired_phrases:
                phrases = phrase_extraction(en_text, fr_text, en2fr)
                if args.filtering:
                    phrases = utils.filtered_phrases(phrases, en_len, fr_len)
                if args.filtering_ngrams: 
                    phrases = utils.filtered_ngrams(phrases, args.max_ngram)
                # print(phrases)
            if args.parse:
                phrases = utils.filter_syntactic_phrases(phrases, syntactic_phrases)
            if args.pos:
                old_phrases = phrases
                phrases = utils.filter_pos_phrases(phrases, en_excluded, fr_excluded)
            if args.missing_phrases:
                en_aligned, fr_aligned = [], []
                for (en_a, fr_a) in en2fr:
                    en_aligned.append(en_a)
                    fr_aligned.append(fr_a)
                en_unaligned = utils.get_unaligned_phrases(en_aligned, en_text)
                fr_unaligned = utils.get_unaligned_phrases(fr_aligned, fr_text)
                for phrase in en_unaligned:
                    phrases.add((phrase[0], None, phrase[1], ''))
                for phrase in fr_unaligned:
                    phrases.add((None, phrase[0], '', phrase[1]))

            score = model([f'{en_text}'], [f'{fr_text}'])
            print(f' >> Original Score: {score[0][0]}')

            divergent_flag = True
            global_phrasal_annotations = [[en_text, fr_text, en_div, fr_div, str(score[0][0])]]
            revisions = 0
            while divergent_flag:

                phrasal_annotations = []
                for phrase in phrases:
                    masked_en = utils.mask_out_phrase(en_text, phrase[2], mask_token=args.mask_token)
                    masked_fr = utils.mask_out_phrase(fr_text, phrase[3], mask_token=args.mask_token)

                    if not masked_en or not masked_fr:
                        continue
                    if utils.punctuation_spans(masked_en) and utils.punctuation_spans(masked_fr):
                        continue

                    masked_score = model([masked_en], [masked_fr])

                    # If masking improved the divergent score, record phrasal pair
                    if masked_score > score:
                        if verbose:
                            print(f'\n{masked_en} ||| {masked_fr}')
                            print(score)
                            print(masked_score)
                        if args.reranking:
                            phrasal_annotations.append([phrase, masked_score - score, masked_score - score])
                        elif args.reward:
                            r = len(masked_en.split(' ')) + len(masked_fr.split(' '))
                            c = len(phrase[2].split(' ')) + len(phrase[3].split(' '))
                            br=np.exp(-c/r)
                            if masked_score[0][0] < 0:
                                phrasal_annotations.append([phrase, [[masked_score[0][0]*(1/br)]], masked_score])
                            else:
                                phrasal_annotations.append([phrase, [[masked_score[0][0]*br]], masked_score])
                        else:
                            phrasal_annotations.append([phrase, masked_score, masked_score])

                if not phrasal_annotations:
                    divergent_flag = False
                    continue
                else:
                    revisions += 1
                    if args.reranking:
                        en_rationale, fr_rationale, score_improvement = utils.constrained_phrase_rationale(phrasal_annotations, en_text, fr_text)
                        #en_rationale, fr_rationale, score_improvement = utils.minimal_phrase_rationale(phrasal_annotations,
                        #                                                                         args.kernel_width)
                        masked_score = score + score_improvement
                    else:
                        en_rationale, fr_rationale, masked_score = utils.phrase_rationale(phrasal_annotations, en_text)
                        if en_rationale=='None' and fr_rationale=='None' and masked_score=='None':
                            break

                    en_text = utils.index_to_text(en_rationale, en_text.split(' '))
                    fr_text = utils.index_to_text(fr_rationale, fr_text.split(' '))
                    score = masked_score

                    en_div_lpo = utils.index_to_div_labels(en_rationale, len(en_text_original.split(' ')))
                    fr_div_lpo = utils.index_to_div_labels(fr_rationale, len(fr_text_original.split(' ')))

                print(f'- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
                print(f'Revision: {revisions}')
                print(f' >> Masked Score: {masked_score}')
                if args.reranking:
                    global_phrasal_annotations.append([en_div_lpo, fr_div_lpo, str(masked_score[0][0])])
                else:
                    global_phrasal_annotations.append([en_div_lpo, fr_div_lpo, str(masked_score)])

            divs_lpo.append(global_phrasal_annotations)

    input_f.close()
    print(f'Writing results to {args.output}')
    with open(args.output, 'w') as output_f:
        for div_lpo in divs_lpo:
            row = []
            revisions = str(len(div_lpo) -1)
            for element in div_lpo[0]:
                row.append(element.rstrip())
            row.append(revisions)
            for revised in div_lpo[1:]:
                row.append(revised[2])
                row.append(revised[0])
                row.append(revised[1])
            row_ = '\t'.join(row)
            output_f.write(f'{row_}\n')
    output_f.close()


if __name__ == "__main__":
    main()
