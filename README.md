# mtanti's PhD
Code used for running experiments for my PhD thesis (link to thesis will be included later). Part of the code was also used for the paper "[Transfer learning from language models to image caption generators: Better models may not transfer better](https://arxiv.org/abs/1901.01216)".

This thesis is an analysis of the different image caption generator neural network architectures available.

Works on Python 3.

## Dependencies

Python dependencies (install all with `pip`):

* `tensorflow==1.4`
* `numpy`
* `scipy`
* `h5py`
* `skopt`
* `nltk`
* `PIL`

## Before running

1. Download Karpathy's [Flickr8K, Flickr30K, and MSCOCO captions](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and put them in `mtanti-phd/datasets/capgen/DATASET/karpathy/dataset.json` where DATASET is flickr8k, flickr30k, or mscoco (rename the files to `dataset.json`!).
1. Download the [Flick8K images](https://forms.illinois.edu/sec/1713398) and put them in `mtanti-phd/datasets/capgen/flickr8k/images`.
1. Download the [Flick30K images](https://forms.illinois.edu/sec/229675) and put them in `mtanti-phd/datasets/capgen/flickr30k/images`.
1. Download the [MSCOCO 2014 images](http://mscoco.org/dataset/#download) and put them all together in `mtanti-phd/datasets/capgen/mscoco/images`.
1. Download [LM1B Google News corpus](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark/archive/master.zip) and extract it in `mtanti-phd/datasets/text/lm1b/1-billion-word-language-modeling-benchmark-master`.
1. Download the [MSCOCO Evaluation toolkit](https://github.com/mtanti/coco-caption/archive/master.zip) extract it in `mtanti-phd/tools/coco-caption-master`.
1. Open `mtanti-phd/experiments/thesis/framework/config/machine_specific.py` and set `base_dir` to the directory of mtanti-php and `val_batch_size` to the maximum batch size that can be processed by your GPU (start with a low number like 100 and keep increasing until you get an out of memory error).
1. Open `mtanti-phd/experiments/thesis/framework/config/general.py` and set `debug` to True or False (True is used to run a quick test).
1. Run `mtanti-phd/experiments/thesis/dataset_maker.py` to pre-process all the data and store it in `mtanti-phd/experiments/thesis/data`.
1. Remove all files inside `mtanti-phd/experiments/thesis/hyperparams` and `mtanti-phd/experiments/thesis/results` as results are not re-computed if already saved.

## To run

All the instructions to run the experiments can be found inside `mtanti-php/experiments/thesis`.