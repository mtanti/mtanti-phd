import os
import sys
import nltk
import collections
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from framework import lib
from framework import data
from framework import config

########################################################################################
if len(sys.argv) == 1:
    architectures = 'merge,par,pre,init'.split(',')
else:
    architectures = sys.argv[1].split(',')

lib.create_dir(config.results_dir+'/imageimportance')

with open(config.results_dir+'/imageimportance/tags.txt', 'w', encoding='utf-8') as f:
    print('architecture', 'dataset', 'run', 'use_generated_sents', 'sent_len', 'token_index', 'tag', 'freq', 'prop', sep='\t', file=f)
    
    for architecture in ['init', 'pre', 'par', 'merge']:
        for dataset_name in ['flickr8k', 'flickr30k', 'mscoco']:
            datasources = data.load_datasources(dataset_name)
            vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq)
            datasources['test'].tokenize_sents()
            
            for use_generated_sents in [ True, False ]:
                for run in range(1, config.num_runs+1):
                    dir_name = '{}_{}_{}'.format(architecture, dataset_name, run)
                    
                    if use_generated_sents == True:
                        with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/sents.txt', 'r', encoding='utf-8') as f2:
                            correct_sents = [ line.split(' ') for line in f2.read().strip().split('\n') ]
                    else:
                        correct_sents = datasources['test'].shuffle(seed=0).take(one_per_group=True).tokenize_sents().text_sents.sents
                    correct_sents = [ vocab.indexes_to_tokens(vocab.tokens_to_indexes(sent)) for sent in correct_sents ]
                        
                    for sent_len in sorted({len(sent) for sent in correct_sents}):
                        print('='*100)
                        print('Sentence length:', sent_len)
                        
                        filtered_correct_sents = list()
                        for correct_sent in correct_sents:
                            if len(correct_sent) == sent_len:
                                filtered_correct_sents.append(correct_sent)
                            
                        print('Number of sentences:', len(filtered_correct_sents))
                        print()
                        if len(filtered_correct_sents) < 20:
                            print('Too little!')
                            print()
                            continue
                        
                        for token_index in range(sent_len):
                            print('-'*100)
                            print(lib.formatted_clock())
                            print(architecture, dataset_name, run, use_generated_sents, sent_len, token_index)
                            print()
                            
                            tag_freqs = collections.defaultdict(lambda:0)
                            for sent in filtered_correct_sents:
                                token_tags = nltk.pos_tag(sent, tagset='universal')
                                tag_freqs[token_tags[token_index][1]] += 1

                            total_freq = sum(tag_freqs.values())
                            for (tag, freq) in tag_freqs.items():
                                print(architecture, dataset_name, run, use_generated_sents, sent_len, token_index, tag, freq, freq/total_freq, sep='\t', file=f)

print(lib.formatted_clock())