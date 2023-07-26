import csv
from collections import defaultdict

from simalign import SentenceAligner

myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

counter = 0
refresd = False
limsi = False
alis = defaultdict(list)
#input_ = '/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/en_fr.devtest'
input_ = '/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/spa_Latn.devtest.divergent'
input_ = '/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/annotations/limsi'
input_ = '/fs/clip-divergences/Ex-QE-data/wmt-qe-2022-data/test_data-gold_labels/task3_ced/pt-en/ced_en_pt'
input_ = '/fs/clip-divergences/Ex-QE-data/mqm/mqm_generalMT2022_en_de'
input_ = '/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/ell_Grek.devtest.equivalents.chatgpt.paraphrase'
input_ = '/fs/clip-divergences/Ex-QE-data/wmt-qe-2022-data/test_data-gold_labels/task3_ced/pt-en/harder/ced_en_pt_with_labels'
#input_ = '/fs/clip-divergences/Ex-QE-data/hallucinations/natural_hallucinations' 
#with open('/fs/clip-divergences/xling-SemDiv/REFreSD/REFreSD_rationale.tsv') as file_:
with open(input_) as file_:
    lines = file_.readlines()
    for id_, line in enumerate(lines):
        l, line = line.split("\t")[0], line.split("\t")[1:]
        if l!='GPT':
            continue
        print(id_)
        #if id_ > 2:
        #    continue
        if refresd:
            print(f'{id_}', end=" ")
            if id_ == 0:
                continue
            lbl = line[1]
            if lbl != 'no_meaning_difference':
                continue

            print(f'{line[2]}\n')
            en_text = line[2].rstrip()
            fr_text = line[3].rstrip()
        elif limsi:
            en_text = line[0].rstrip()
            fr_text = line[1].rstrip()

        else:
            en_text = line[0].rstrip()
            fr_text = line[1].rstrip()
        alignments = myaligner.get_word_aligns(en_text, fr_text)
        for matching_method in alignments:
            if refresd:
                alis[matching_method].append([en_text, fr_text, line[4].rstrip(), line[5].rstrip(), alignments[matching_method]])
            elif limsi: 
                alis[matching_method].append([en_text, fr_text, line[2].rstrip(), line[3].rstrip(), alignments[matching_method], line[4].rstrip()])
            else:
                #alis[matching_method].append([en_text, fr_text, line[2].rstrip(), line[3].rstrip(), alignments[matching_method]])
                alis[matching_method].append([en_text, fr_text, alignments[matching_method]])

for matching_method in alis.keys():
    with open(f'{input_}_{matching_method}', 'w') as f_output:
        for line in alis[matching_method]:
            f_output.write(f'{line[0]}\t{line[1]}\t{line[2]}\n')            
            #f_output.write(f'{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t{line[4]}\n')
            #f_output.write(f'{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t{line[4]}\t{line[5]}\n')
   
            
