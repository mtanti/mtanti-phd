#In whereimage_experiment.py line 176, I mistakenly shuffled all the test sets, which I forgot was a mutable operation, leading to result files where each line corresponds to an image (wmd.txt, sents.txt, etc.) are out of sync with the actual image order in the dataset. This script generates the order of the shuffled indexes in order to allow for reordering.
from framework import lib
from framework import data
from framework import config

for dataset_name in ['flickr8k', 'flickr30k', 'mscoco']:
    for architecture in ['ceiling', 'init', 'pre', 'par', 'merge']:
        for run in range(1, config.num_runs+1):
            dir_name = '{}_{}_{}'.format(architecture, dataset_name, run)
            datasources = data.load_datasources(dataset_name)
            datasources['test'].shuffle(run)
            with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/shuffled_test_indexes.txt', 'w', encoding='utf-8') as f:
                for index in datasources['test'].take(one_per_group=True).group_indexes:
                    print(index, file=f)
