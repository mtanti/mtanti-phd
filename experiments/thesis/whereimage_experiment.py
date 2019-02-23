import os
import numpy as np
import shutil
import sys
import contextlib
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from framework import lib
from framework import model_neural_trad
from framework import evaluation
from framework import data
from framework import config

########################################################################################
class _FitListener(model_neural_trad.FitListener):
    
    def __init__(self):
        super(_FitListener, self).__init__()
        self.epoch_timer = None
        self.training_prog = None
    
    def epoch_started(self, model, epoch_num):
        print(epoch_num, end='\t')
        self.epoch_timer = lib.Timer()
    
    def minibatch_ready(self, model, items_ready, num_items):
        if self.training_prog is None:
            self.training_prog = lib.ProgressBar(num_items, 5)
        self.training_prog.inc_value()
    
    def epoch_ready(self, model, epoch_num, train_logpplx, val_logpplx):
        if epoch_num == 0:
            print(' '*lib.ProgressBar.width(5), end=' | \t')
        else:
            print(' | ', end='\t')
        print(round(train_logpplx, 3), round(val_logpplx, 3), lib.format_duration(self.epoch_timer.get_duration()), sep='\t')
        self.training_prog = None
        
        
########################################################################################
if len(sys.argv) == 1:
    architectures = 'ceiling,merge,par,pre,init,merge-ext'.split(',')
else:
    architectures = sys.argv[1].split(',')

lib.create_dir(config.results_dir+'/whereimage')

for architecture in architectures:
    lib.create_dir(config.results_dir+'/whereimage/'+architecture)
    if not lib.file_exists(config.results_dir+'/whereimage/'+architecture+'/results.txt'):
        with open(config.results_dir+'/whereimage/'+architecture+'/results.txt', 'w', encoding='utf-8') as f:
            print(
                    'architecture',
                    'dataset',
                    'run',
                    'vocab_size',
                    'num_params',
                    'mean_prob',
                    'median_prob',
                    'geomean_prob',
                    'mean_pplx',
                    'median_pplx',
                    'geomean_pplx',
                    'vocab_used',
                    'vocab_used_frac',
                    'min_freq_vocab_used',
                    'min_sent_len',
                    'mean_sent_len',
                    'max_sent_len',
                    'num_reused_sents',
                    'num_reused_sents_frac',
                    'reused_sents_WMD',
                    'num_reused_3grams',
                    'num_reused_4grams',
                    'num_reused_5grams',
                    'num_unique_sents',
                    'num_unique_sents_frac',
                    'num_types_nouns',
                    'num_types_adjectives',
                    'num_types_verbs',
                    'num_types_adverbs',
                    'num_types_unigrams',
                    'num_types_bigrams',
                    'num_types_trigrams',
                    'BLEU_1',
                    'BLEU_2',
                    'BLEU_3',
                    'BLEU_4',
                    'METEOR',
                    'ROUGE_L',
                    'CIDEr',
                    'SPICE',
                    'WMD',
                    'R@1',
                    'R@5',
                    'R@10',
                    'median_rank',
                    'R@1_frac',
                    'R@5_frac',
                    'R@10_frac',
                    'median_rank_frac',
                    'num_epochs',
                    'training_time',
                    'total_time',
                    sep='\t', file=f
                )
    already_seen = set()
    with open(config.results_dir+'/whereimage/'+architecture+'/results.txt', 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n')[1:]:
            [
                architecture,
                dataset,
                run,
            ] = line.split('\t')[:3]
            
            next_exp = [
                architecture,
                dataset,
                int(run),
            ]
            
            already_seen.add(tuple(next_exp))

    if architecture != 'ceiling':
        with open(config.hyperpar_dir+'/whereimage/'+architecture+'/best.txt', 'r', encoding='utf-8') as f:
            hyperpars = dict(line.split('\t') for line in f.read().strip().split('\n'))
        init_method             = hyperpars['init_method']
        max_init_weight         = float(hyperpars['max_init_weight'])
        embed_size              = int(hyperpars['embed_size'])
        rnn_size                = int(hyperpars['rnn_size'])
        post_image_size         = int(hyperpars['post_image_size']) if hyperpars['post_image_size'] != 'None' else None
        pre_output_size         = int(hyperpars['pre_output_size']) if hyperpars['pre_output_size'] != 'None' else None
        post_image_activation   = hyperpars['post_image_activation']
        rnn_type                = hyperpars['rnn_type']
        optimizer               = hyperpars['optimizer']
        learning_rate           = float(hyperpars['learning_rate'])
        normalize_image         = hyperpars['normalize_image'] == 'True'
        weights_reg_weight      = float(hyperpars['weights_reg_weight'])
        image_dropout_prob      = float(hyperpars['image_dropout_prob'])
        post_image_dropout_prob = float(hyperpars['post_image_dropout_prob'])
        embedding_dropout_prob  = float(hyperpars['embedding_dropout_prob'])
        rnn_dropout_prob        = float(hyperpars['rnn_dropout_prob'])
        max_gradient_norm       = float(hyperpars['max_gradient_norm'])
        minibatch_size          = int(hyperpars['minibatch_size'])
        beam_width              = int(hyperpars['beam_width'])
    
    for dataset_name in ['flickr8k', 'flickr30k', 'mscoco']:
        for run in range(1, config.num_runs+1):
            print('='*100)
            print(lib.formatted_clock())
            print(architecture, dataset_name, run)
            print()
            
            if (architecture, dataset_name, run) in already_seen:
                print('Found ready')
                print()
                continue
            
            full_timer = lib.Timer()
            
            dir_name = '{}_{}_{}'.format(architecture, dataset_name, run)
            lib.create_dir(config.results_dir+'/whereimage/'+architecture+'/'+dir_name)
            
            datasources = data.load_datasources(dataset_name)
            vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq)
            
            dataset = data.Dataset(
                    vocab            = vocab,
                    train_datasource = datasources['train'],
                    val_datasource   = datasources['val'],
                    test_datasource  = datasources['test'],
                )
            dataset.compile_sents()
            
            selected_test_sents = dataset.test.shuffle(run).take(one_per_group=True).tokenize_sents().compile_sents(vocab)
            selected_index_sents = selected_test_sents.index_sents
            
            with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/selected_test.txt', 'w', encoding='utf-8') as f:
                print(*selected_test_sents.individual_indexes, sep='\n', file=f)
            
            vocab.save_vocab(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/vocab.json')
            
            if architecture != 'ceiling':
                model = model_neural_trad.TradNeuralModel(
                        vocab_size              = vocab.size,
                        init_method             = init_method,
                        max_init_weight         = max_init_weight,
                        embed_size              = embed_size,
                        rnn_size                = rnn_size,
                        post_image_size         = post_image_size,
                        pre_output_size         = pre_output_size,
                        post_image_activation   = post_image_activation,
                        rnn_type                = rnn_type,
                        architecture            = architecture,
                        optimizer               = optimizer,
                        learning_rate           = learning_rate,
                        normalize_image         = normalize_image,
                        weights_reg_weight      = weights_reg_weight,
                        image_dropout_prob      = image_dropout_prob,
                        post_image_dropout_prob = post_image_dropout_prob,
                        embedding_dropout_prob  = embedding_dropout_prob,
                        rnn_dropout_prob        = rnn_dropout_prob,
                        max_gradient_norm       = max_gradient_norm,
                        freeze_prefix_params    = False,
                    )
            else:
                model = contextlib.suppress()
                
            with model:
                if architecture != 'ceiling':
                    model.compile_model()
                    model.init_params()
                
                train_timer = lib.Timer()
                
                print('Training...')
                
                if architecture != 'ceiling':
                    fit_stats = model.fit(dataset, config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/model.hdf5', max_batch_size=config.val_batch_size, minibatch_size=minibatch_size, max_epochs=config.max_epochs, early_stop_patience=config.early_stop_patience, listener=_FitListener())
                    with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/train.txt', 'w', encoding='utf-8') as f:
                        print('epoch', 'train logpplx', 'val logpplx', sep='\t', file=f)
                        for (epoch, train_logpplx, val_logpplx) in zip(range(fit_stats['num_epochs']+1), fit_stats['train_logpplxs'], fit_stats['val_logpplxs']):
                            print(epoch, train_logpplx, val_logpplx, sep='\t', file=f)
                else:
                    fit_stats = {'num_epochs': 0}
                print()
                
                train_duration = train_timer.get_duration()
                
                print('Probability stats...')
                print()
                
                if architecture != 'ceiling':
                    (sents_logprobs, tokens_logprobs) = model.get_sents_logprobs(max_batch_size=config.val_batch_size, index_sents=dataset.test.index_sents, images=dataset.test.images)
                    prob_stats = evaluation.get_probability_stats(sents_logprobs, dataset.test.index_sents.lens)
                    with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/probs.txt', 'w', encoding='utf-8') as f:
                        for logprobs in tokens_logprobs:
                            print(*logprobs, sep='\t', file=f)
                else:
                    prob_stats = {'mean_prob': -1, 'median_prob': -1, 'geomean_prob': -1, 'mean_pplx': -1, 'median_pplx': -1, 'geomean_pplx': -1}
                
                print('Generating sentences...')
                
                prog = lib.ProgressBar(dataset.test.num_groups, 5)
                if architecture != 'ceiling':
                    (index_sents, logprobs) = model.generate_sents_beamsearch(max_batch_size=config.val_batch_size, images=dataset.test.get_images(), beam_width=beam_width, lower_bound_len=config.lower_bound_len, upper_bound_len=config.upper_bound_len, temperature=config.temperature, listener=lambda num_ready:prog.update_value(num_ready))
                    text_sents = index_sents.decompile_sents(vocab).sents
                else:
                    rand = random.Random(run)
                    text_sents = [rand.choice(group) for group in dataset.test.get_text_sent_groups()]
                    
                with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/sents.txt', 'w', encoding='utf-8') as f:
                    for sent in text_sents:
                        print(*sent, sep=' ', file=f)
                print()
                print()
                
                print('Diversity stats...')
                print()
                
                diversity_stats = evaluation.diversity_eval(dataset.train.text_sents.sents, dataset.test.get_text_sent_groups(), vocab, dataset.train.text_sents.get_vocab_freqs(), text_sents)
                
                print('Quality stats...')
                print()
                
                generation_stats = evaluation.mscoco_eval(dataset.test.get_text_sent_groups(), text_sents)
                with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/wmd.txt', 'w', encoding='utf-8') as f:
                    print(*generation_stats['WMD_all'], sep='\n', file=f)
                print()
                
                print('Generating image-caption probability matrix...')
                
                if architecture != 'ceiling':
                    prog = lib.ProgressBar(dataset.test.num_groups**2, 5)
                    image_caption_logprobs_matrix = list()
                    for (sent_prefix, sent_len, sent_target) in zip(selected_index_sents.prefixes, selected_index_sents.lens, selected_index_sents.targets):
                        index_sents = data.IndexSents(np.array([sent_prefix]*selected_index_sents.size, np.int32), np.array([sent_len]*selected_index_sents.size, np.int32), np.array([sent_target]*selected_index_sents.size, np.int32))
                        logprobs = model.get_sents_logprobs(max_batch_size=config.val_batch_size, index_sents=index_sents, images=dataset.test.get_images(), listener=lambda num_ready:prog.inc_value())[0]
                        image_caption_logprobs_matrix.append(logprobs)
                print()
                print()
                
                print('Retrieval stats...')
                print()
                
                if architecture != 'ceiling':
                    retrieval_stats = evaluation.retrieval_eval(image_caption_logprobs_matrix)
                    with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/rets.txt', 'w', encoding='utf-8') as f:
                        for row in image_caption_logprobs_matrix:
                            print(*row, sep='\t', file=f)
                else:
                    retrieval_stats = {'R@1': -1, 'R@5': -1, 'R@10': -1, 'median_rank': -1, 'R@1_frac': -1, 'R@5_frac': -1, 'R@10_frac': -1, 'median_rank_frac': -1}
                
                if architecture != 'ceiling':
                    num_params = model.get_num_params()
                else:
                    num_params = 0
                    
            full_duration = full_timer.get_duration()
            
            print('Done!')
            
            with open(config.results_dir+'/whereimage/'+architecture+'/results.txt', 'a', encoding='utf-8') as f:
                print(
                    architecture,
                    dataset_name,
                    run,
                    vocab.size,
                    num_params,
                    prob_stats['mean_prob'],
                    prob_stats['median_prob'],
                    prob_stats['geomean_prob'],
                    prob_stats['mean_pplx'],
                    prob_stats['median_pplx'],
                    prob_stats['geomean_pplx'],
                    diversity_stats['vocab_used'],
                    diversity_stats['vocab_used_frac'],
                    diversity_stats['min_freq_vocab_used'],
                    diversity_stats['min_sent_len'],
                    diversity_stats['mean_sent_len'],
                    diversity_stats['max_sent_len'],
                    diversity_stats['num_reused_sents'],
                    diversity_stats['num_reused_sents_frac'],
                    diversity_stats['reused_sents_WMD'],
                    diversity_stats['num_reused_3grams'],
                    diversity_stats['num_reused_4grams'],
                    diversity_stats['num_reused_5grams'],
                    diversity_stats['num_unique_sents'],
                    diversity_stats['num_unique_sents_frac'],
                    diversity_stats['num_types_nouns'],
                    diversity_stats['num_types_adjectives'],
                    diversity_stats['num_types_verbs'],
                    diversity_stats['num_types_adverbs'],
                    diversity_stats['num_types_unigrams'],
                    diversity_stats['num_types_bigrams'],
                    diversity_stats['num_types_trigrams'],
                    generation_stats['Bleu_1'],
                    generation_stats['Bleu_2'],
                    generation_stats['Bleu_3'],
                    generation_stats['Bleu_4'],
                    generation_stats['METEOR'],
                    generation_stats['ROUGE_L'],
                    generation_stats['CIDEr'],
                    generation_stats['SPICE'],
                    generation_stats['WMD'],
                    retrieval_stats['R@1'],
                    retrieval_stats['R@5'],
                    retrieval_stats['R@10'],
                    retrieval_stats['median_rank'],
                    retrieval_stats['R@1_frac'],
                    retrieval_stats['R@5_frac'],
                    retrieval_stats['R@10_frac'],
                    retrieval_stats['median_rank_frac'],
                    fit_stats['num_epochs'],
                    train_duration,
                    full_duration,
                    sep='\t', file=f
                )
            
            already_seen.add((architecture, dataset_name, run))
            print()
            print(lib.format_duration(full_duration))
            print()
                
print(lib.formatted_clock())