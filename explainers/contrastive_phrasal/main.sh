# French -- ChatGPT paraphrase
output=${output_dir}/fra_Latn.devtest.equivalents.chatgpt.paraphrase_missing_paired_phrases_reward
input=/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/fra_Latn.devtest.equivalents.chatgpt.paraphrase_inter
model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
python main.py --paired_phrases --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}

# French -- ChatGPT paraphrase
output=${output_dir}/spa_Latn.devtest.equivalents.chatgpt.paraphrase_missing_paired_phrases_reward
input=/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/spa_Latn.devtest.equivalents.chatgpt.paraphrase_inter
model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert_new/from_WikiMatrix.en-es.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
python main.py --paired_phrases --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}


#output=${output_dir}/ell_Grek.devtest.equivalents.chatgpt.paraphrase_missing_paired_phrases_reward
#input=/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/ell_Grek.devtest.equivalents.chatgpt.paraphrase_inter
#model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.el-en.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
#python main.py --paired_phrases --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}

#exit

# Portuguese CED
output=${output_dir}/en_pt_ced_test_with_labels_missing_paired_phrases_reward_pos
input=${input_dir}/ced_en_pt_with_labels_itermax_pos_pos
model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.en-pt.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
python main.py --paired_phrases --pos --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}
exit


# Greek FLORES
output=${output_dir}/ell_Grek.devtest_missing_paired_phrases_reward
input=/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/ell_Grek.devtest.divergent_itermax
model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.el-en.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
python main.py --paired_phrases --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}


# French FLORES
output=${output_dir}/fra_Latn.devtest_missing_paired_phrases_reward
input=/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/fra_Latn.devtest.divergent_itermax
model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert/from_WikiMatrix.en-fr.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
python main.py --paired_phrases --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}


# Spanish FLORES
output=${output_dir}/spa_Latn.devtest_missing_paired_phrases_reward
input=/fs/clip-divergences/xling-SemDiv/Ex-SemDiv/flores200_dataset/devtest/spa_Latn.devtest.divergent_itermax
model_name_or_path=/fs/clip-divergences/xling-SemDiv/trained_bert_new/from_WikiMatrix.en-es.tsv.filtered_sample_50000.moses.seed/contrastive_divergence_ranking/rdpg
python main.py --paired_phrases --missing_phrases --model_name_or_path ${model_name_or_path} --reward --input ${input} --output ${output}

# REFreSD
#output=${output_dir}/sd_missing_paired_pos_phrases_reward
#input=${input_dir}/sd_inter_pos_both
#python main.py --paired_phrases --missing_phrases --pos --reward --input ${input} --output ${output}
