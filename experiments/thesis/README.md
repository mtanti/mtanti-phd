## Instructions for running experiments

* For running experiments in Chapter 3 of thesis (Conditioned language models):
1. Run `whereimage_hyperpar.py` to find best hyperparameters for each caption generation architecture. Hyperparameters are stored in `hyperparams/whereimage`.
1. Run `whereimage_experiment.py` to measure the performance of each architecture using the hyperparameters found in the previous step. Results are stored in `results/whereimage`.
1. Run `whereimage_get_eval_sample.py` to generate sample data for use in human evaluation website. Results are stored in `results/whereimage/_sample`. Results of actual human evaluation are stored in `results/whereimage/_sample/humaneval_export.txt`.

* For running experiments in Chapter 4 of thesis (Groundedness analysis):
1. Run `imageimportance_get_foils.py` to get foil images for use in omission score experiments. Results are stored in `results/imageimportance`.
1. Run `imageimportance_get_partsofspeech.py` to get part of speech frequencies for different test set caption positions. Results are stored in `results/imageimportance`.
1. Run `imageimportance_experiment.py` to measure the sensitivity and omission scores of the different architectures. Results are stored in `results/imageimportance`.
1. Run `imageimportance_cosine_randoms.py` to measure the omission score of random vectors. Results are displayed.
1. Run `imageimportance_logits.py` to get statistics on the logits of the softmax. Results are stored in `results/imageimportance`.

* For running experiments in Chapter 5 of thesis (Transfer learning using the merge architecture) as well as in the paper "Transfer learning from language models to image caption generators: Better models may not transfer better":
1. Run `langmodtrans_hyperpar.py` to find the best hyperparameters for each language model and target caption generator. Results are stored in `hyperparams/langmodtrans`.
1. Run `langmodtrans_experiment.py` to measure the performance of the merge architecture after transfer learning from different language models. Results are stored in `results/langmodtrans`.
1. Run `langmodtrans_val_pplx.py` to measure the perplexity of the language models. Results are stored in `results/langmodtrans`.
1. Run `partialtraining_experiment_earlystop.py` to measure the performance of the merge architecture after transfer learning from partially trained language models. Results are stored in `results/partialtraining`.
1. Run `partialtraining_experiment_noearlystop.py` to do the same thing that `partialtraining_experiment_earlystop.py` does but does not use early stopping. Continues from where `partialtraining_experiment_earlystop.py` left off. Results are stored in `results/partialtraining`.
1. Run `randomrnn_experiment.py` to measure the performance of the merge caption generator when its embedding layer and RNN have random parameters and using the manually set hyperparameters in `hyperparams/randomrnn`. Results are stored in `results/randomrnn`.

* After running `langmodtrans_experiment.py`, you can then run `imageimportance_langmod.py` to measure the sensitivity of language models. Results are stored in `results/imageimportance`.