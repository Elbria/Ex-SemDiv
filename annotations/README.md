# Alignment-augmented Bitexts

This directory contains bitexts, augmented with word-alignment annotations and (optionally) part-of-speech tags, used as inputs by the contrastive explainer.

The datasets covered include: 

- [x] English-French dataset used in our *proxy evaluations*.
* ``some_meaning_difference_alignments_pos``: English-French parallel texts containing fine-grained meaning divergences, found in [ReFreSD](https://github.com/Elbria/xling-SemDiv/tree/master/REFreSD/REFreSD_for_huggingface).
- [x] English-French and English-Spanish datasets used in the "Annotation of Semantic Divergences" application-grounded evaluation:
* ``*_Latn.devtest.equivalents``: English-{French, Spanish} parallel texts from [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md). This file contains 67 parallel texts detected as equivalent under the [divergentmBERT model](https://github.com/Elbria/xling-SemDiv) for both French and Spanish. 
* ``*_Latn.devtest.equivaelnts..chatgpt.edits_alignments``: English-{French-Spanish} versions of ``*_Latn.devtest.equivalents`` where part of the English-text has been minimally edited with ChatGPT to *deliberately introduce meaning differences*.
* ``*_Latn.devtest.equivaelnts..chatgpt.paraphrase_alignments``: English-{French-Spanish} versions of ``*_Latn.devtest.equivalents`` where part of the English-text has been minimally edited with ChatGPT to *deliberately introduce syntactic differences*.
- [x] English-Portuguese dataset used in the "Critical Error Detection" application-grounded evaluation.
* ``ced_en_pt_alignments_pos``: English-Portuguese MT texts from the [WMT 2022 Task on Critical Error Detection](test_data-gold_labels/task3_ced/pt-en).
* ``ced_en_pt.chatgpt.edits_alignments_pos``: Sample of ``ced_en_pt_alignments_pos`` containing *negation* errors introduced by editing the English side of the parallel texts, with ChatGPT.