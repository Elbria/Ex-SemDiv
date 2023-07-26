import re
import string
import numpy as np
punctuation_types = [punc for punc in string.punctuation]
excluded_types = punctuation_types + ['PUNC', '-LRB-', '-RRB-', 'DT', 'IN', '``', 'HYPH', 'CC', 'TO', 'PUNCT', 'DET', 'CCONJ', 'ADP' ]


def ali_to_tuple(alis):
    """"Converts string of alignment to tuple"""
    en2fr = []
    alis = alis.split('), (')
    for ali_ in alis:
        ali_ = re.sub(r'[^\w\s]', '', ali_).split(' ')
        en2fr.append((int(ali_[0]), int(ali_[1])))
    return en2fr


def mask_out_phrase_old(text, phrase):
    """Masks out a phrase from text"""
    mask = '' * len(phrase.split(' '))
    new_text = f'{text[:text.index(phrase)]}{mask}{text[text.index(phrase) + len(phrase):]}'
    new_text = re.sub(' +', ' ', new_text).rstrip().lstrip().capitalize()
    if new_text:
        if new_text[-1] in string.punctuation:
            return new_text
        new_text = f'{new_text} .'
    return new_text


def mask_out_phrase(text, phrase, mask_token=False):
    """Masks out a phrase from text"""
    if mask_token:
        mask = ' [MASK] ' * len(phrase.split(' '))
    else:
        mask = '' * len(phrase.split(' '))
    try:
        new_text = f'{text[:text.index(phrase)]}{mask}{text[text.index(phrase)+len(phrase):]}'
        new_text = re.sub(' +', ' ', new_text).rstrip().lstrip()
        new_text = ''.join(new_text[:1].upper() + new_text[1:])
        if new_text:
            if new_text[-1] in string.punctuation:
                if new_text[-1] == ',':
                    new_text = ''.join(new_text[:-1] + '.')
                return new_text
            new_text = f'{new_text} .'
        return new_text
    except ValueError:
        return None

def punctuation_spans(text):
    """Checks if text contains only punctuation symbols"""
    punctuation_flag = True
    for t in text:
        if t not in string.punctuation and t != ' ':
            punctuation_flag = False
    return punctuation_flag


def phrase_rationale(phrasal_annotations, en_text):
    """"Extracts one best phrasal rational from candidates"""
    max_change = -10000000
    for ann in phrasal_annotations:
        if ann[1][0][0] > max_change and ann[0][2] != en_text:
            max_ann = ann
            max_change = ann[1][0][0]
    try:
        return max_ann[0][0], max_ann[0][1], max_ann[2][0][0]
    except UnboundLocalError as error:
        return 'None', 'None', 'None'

def index_to_div_labels(ind, len_):
    """Extracts divergent labels from phrasal indices"""
    if not ind:
        return ' '.join(['0']*len_)
    labels = []
    for i in range(len_):
        if i >= ind[0] and i < ind[1]:
            labels.append('1')
        else:
            labels.append('0')
    return ' '.join(labels)


def index_to_text(ind, text):
    """Extracts texts from phrasal indices"""
    if not ind:
        return ' '.join(text)
    tokens = []
    for i, token in enumerate(text):
        if i >= ind[0] and i < ind[1]:
            continue
        else:
            tokens.append(token)
    return ' '.join(tokens)


def get_unaligned_phrases(aligned_indices, text):
    unaligned_phrase, unaligned_ind,  unaligned_phrases = [], [], []
    for i in range(len(text.split(' '))):
        if i not in aligned_indices:
            unaligned_phrase.append(text.split(' ')[i])
            unaligned_ind.append(i)
        else:
            if unaligned_phrase and len(unaligned_phrase) > 1:
                unaligned_phrases.append([(unaligned_ind[0], unaligned_ind[-1] +1), ' '.join(unaligned_phrase)])
            unaligned_phrase = []
            unaligned_ind = []
    return unaligned_phrases


def minimal_phrase_rationale(phrasal_annotations, kernel_width):
    max_change = -1000000
    for ann in phrasal_annotations:
        d = min(len(ann[0][2].split(' ')), len(ann[0][3].split(' ')))
        weighted_score = float(ann[1][0][0]) * np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        if weighted_score > max_change:
            max_ann = ann
            max_change = weighted_score
    return max_ann[0][0], max_ann[0][1], max_ann[1][0][0]


def constrained_phrase_rationale(phrasal_annotations, en_text, fr_text):
    
    min_phrase = 100
    for ann in phrasal_annotations:
  
        #d_phrase = len(ann[0][2].split(' ')) + len(ann[0][3].split(' '))
        d_phrase = min(len(ann[0][2].split(' ')) , len(ann[0][3].split(' ')))
        d_sent = len(en_text.split(' ')) + len(fr_text.split(' '))        
        
        if d_phrase < min_phrase:# and ann[0][2] != en_text:
            min_ann = ann
            min_phrase = d_phrase
    return min_ann[0][0], min_ann[0][1], min_ann[1]

def filtered_phrases(phrases, en_len, fr_len):
    filtered_phrases_ =  set()
    for phrase in phrases:
        if len(phrase[2].split(' ')) > en_len or len(phrase[2].split(' ')) > fr_len:
            continue
        filtered_phrases_.add(phrase)
    return filtered_phrases_


def filtered_ngrams(phrases, max_n_gram):
    filtered_phrases_ =  set()
    for phrase in phrases:
        if len(phrase[2].split(' ')) >= max_n_gram and len(phrase[3].split(' ')) >= max_n_gram:
            continue
        filtered_phrases_.add(phrase)
    return filtered_phrases_

def extract_syntactic_phrases(text, mappings):
    phrases = []
    mappings = mappings.split(' ')
    for mapping in mappings:
        mapping = mapping.split(':')
        phrases.append(text[int(mapping[0]):int(mapping[1])])
    return phrases

def filter_syntactic_phrases(entire_set, syntactic_list):
    phrases = set()
    for element in entire_set:
        if element[2] not in syntactic_list:
            continue
        phrases.add(element)
    return phrases

def filter_pos_phrases(entire_set, en_list, fr_list):
    phrases = set()
    for element in entire_set:
        #print(element)
        if element[2] in en_list and element[3] in fr_list:
            continue
        if element[2] == element[3]:
            continue
        phrases.add(element)
    return phrases  

def extract_invalid_pos(en, en_pos, fr, fr_pos):
    en_exclude, fr_exclude = [], []
    for e, p in zip(en.split(' '), en_pos.split(' ')):
        if p in excluded_types:
            en_exclude.append(e)
            
    for e, p in zip(fr.split(' '), fr_pos.split(' ')):
        if p in excluded_types:
            fr_exclude.append(e)
    
    return set(en_exclude), set(fr_exclude)
