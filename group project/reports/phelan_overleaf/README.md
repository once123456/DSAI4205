# Phelan's English model report

Upload the Overleaf ZIP as a new project and compile `main.tex`. The source includes `phelan_section.tex`, `transfer_study.tex`, `references.tex` and `transfer_class_f1.png`. To integrate into an existing group report, copy both section files and the figure and merge the bibliography entries.

We compared 11 configurations. The locked challenger used base BERT with text alone. On the shared 397-message test set it scored 0.4686 macro-F1 and 0.4786 accuracy, against 0.4971 and 0.5113 for the previous class-weight model. No finalist met all three validation thresholds. The class-weight configuration (v4) remains the selected submission model and the challenger is an exploratory result.

Both translation directions ran locally on the 121 Neutral training parents. The fixed checks accepted 76 proposed paraphrases before fold-specific deduplication. Across the 473 pooled out-of-fold predictions, macro-F1 was 0.4557 without augmentation, 0.4599 with matched repetition and 0.4539 with back-translation. Back-translation changed Neutral F1 by +0.0025 versus matched repetition.

Run the second study from its setup cell in `../../notebooks/phelan/BERT_MLP_Refinement.ipynb`. The notebook lists package versions and pinned model identifiers. The paths assume the shared processed data and saved comparator predictions. Each run creates a separate results directory. The published evidence is in `../../runs/phelan/20261121_transfer_study/`. Earlier class-weight and head experiments are preserved in `../../runs/phelan/20261121_refinement_v4/`.

The Word and PDF exports cover this individual contribution. The group report and integrated group notebook require the remaining members' verified artifacts.
