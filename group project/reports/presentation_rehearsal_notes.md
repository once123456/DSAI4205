# Presentation Rehearsal Notes

Working draft, 16 November 2026. The first 12 slides are allocated 12 minutes in total. Slide 13 is a Q&A appendix. Presenter names, team results, and the final timing need to be confirmed after the remaining models arrive. The final comparison may require replacing or adding slides while preserving the 12-minute limit.

| Slide | Time | Presenter | Talking points |
|---|---:|---|---|
| 1 | 0:35 | To confirm | We study whether a four-class classifier can help a team review Twitter messages. We focus on data quality, comparable evaluation, and where the model still needs human judgment. |
| 2 | 0:50 | To confirm | Explain the four labels and the raw 596/397 train/test rows. The files do not have a separate target-entity field, which matters when one message mentions several brands. |
| 3 | 0:55 | To confirm | The cleaned training pool contains 591 rows. Negative is the largest class at 174, while Irrelevant is the smallest at 106. Macro-F1 helps show performance across all four classes. |
| 4 | 1:00 | To confirm | Explain duplicate and overlap handling: four redundant training copies and one overlapping training row were removed. The test copy was retained and flagged. The fixed development split is 473 training and 118 validation rows. |
| 5 | 0:55 | To confirm | In the training pool, question marks appear in 25.9% of Negative and 9.1% of Positive messages. These are descriptive associations, not proof of causation. Four surface counts were later included in BERT+MLP. |
| 6 | 1:05 | To confirm | Describe frozen BERT as the feature extractor: mean pooling gives 768 features, augmented by four standardized counts. A separate MLP predicts the labels. Explain that remaining required model pairs are pending and that Qwen is reported as an exploratory method. |
| 7 | 1:15 | To confirm | On the same 397 test rows and using the 473-row training subset, original TF-IDF+LR scores 0.4461 macro-F1 and BERT+MLP scores 0.4843, an observed difference of 3.82 percentage points. The validation-selected baseline scores 0.4368. Do not imply these differences are statistically established. |
| 8 | 1:05 | To confirm | The baseline Irrelevant recall is 0.1515, compared with 0.3333 for BERT. Its precision falls from 0.8333 to 0.2857. Which trade-off is acceptable depends on the cost of missed cases and false alarms. |
| 9 | 1:20 | To confirm | Qwen's standalone label-only result is 0.4850. The rationale experiment must be compared with its own control: the observed difference is +0.0141, with a 95% interval from −0.0195 to 0.0508. The counterfactual experiment's own comparison gives +0.0017, interval −0.0408 to 0.0471. Both intervals cross zero, so neither experiment establishes a reliable gain. |
| 10 | 0:55 | To confirm | The baseline refinement selected among 24 configurations using validation macro-F1. The selected configuration scored 0.4368 on test versus 0.4461 for the original baseline. Selection on validation does not guarantee improved performance on new data. |
| 11 | 0:55 | To confirm | A practical use is to prioritize messages for review, while checking both the original text and the entity. Small sample size, disputed labels, repeated test inspection and missing entity annotations limit deployment claims. |
| 12 | 1:10 | To confirm | Replace this placeholder slide with the completed team comparison and actual contribution table. Finish with the final model's observed results, uncertainty, and a concrete next step. Rebalance timing once new results are available. |
| 13 | Outside 12:00 | To confirm | Backup Q&A. Potential questions: Why is Irrelevant difficult? Why do rationale gains remain uncertain? How was test leakage controlled, and what evidence would a new dataset provide? |

Total for slides 1–12: **12:00**. Rehearse aloud and leave room for transitions; the current times are estimates, not a measured recording.
