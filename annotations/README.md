# Alignment Augmented Bitexts

This directory contains bitexts, augmented with word-alignment annotations and (optionally) part-of-speech tags.

The datasets covered include: 

* ``some_meaning_difference_alignments_pos``: English-French parallel texts containing fine-grained meaning divergences, found in [ReFreSD](https://github.com/Elbria/xling-SemDiv/tree/master/REFreSD/REFreSD_for_huggingface).
* ``*_Latn.devtest.equivalents``: English-{French, Spanish} parallel texts from [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md). This file contains 67 parallel texts detected as equivalent under the [divergentmBERT model](https://github.com/Elbria/xling-SemDiv) for both French and Spanish. 
* ``*_Latn.devtest.equivaelnts..chatgpt.edits_alignments``: English-{French-Spanish} versions of ``*_Latn.devtest.equivalents`` where part of the English-text has been minimally edited with ChatGPT to *deliberately introduce meaning differences*.
* * ``*_Latn.devtest.equivaelnts..chatgpt.paraphrase_alignments``: English-{French-Spanish} versions of ``*_Latn.devtest.equivalents`` where part of the English-text has been minimally edited with ChatGPT to *deliberately introduce syntactic differences*.