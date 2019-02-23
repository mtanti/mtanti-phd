#In langmodtrans_experiment.py line 244, I mistakenly shuffled all the test sets, which I forgot was a mutable operation, leading to result files where each line corresponds to an image (wmd.txt, sents.txt, etc.) are out of sync with the actual image order in the dataset. This script generates the order of the shuffled indexes in order to allow for reordering.
from framework import lib
from framework import data
from framework import config

for run in range(1, config.num_runs+1):
    datasources = data.load_datasources(config.langmodtrans_capgen_dataset)
    datasources['test'].shuffle(run)
    for corpus in ['lm1b', 'mscoco', 'flickr8k']:
        for frozen_prefix in [True, False]:
            for corpus_size_factor_exponent in (config.langmodtrans_corpus_size_factor_exponents if corpus != config.langmodtrans_capgen_dataset else config.langmodtrans_corpus_size_factor_minor_exponents):
                dir_name = '{}_{}_{}'.format(frozen_prefix, corpus_size_factor_exponent, run)
                with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/shuffled_test_indexes.txt', 'w', encoding='utf-8') as f:
                    for index in datasources['test'].take(one_per_group=True).group_indexes:
                        print(index, file=f)
