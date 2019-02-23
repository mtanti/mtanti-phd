import os
import numpy as np
import shutil
import sys
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
    corpora = 'lm1b,mscoco,flickr8k'.split(',')
else:
    corpora = sys.argv[1].split(',')

datasources = data.load_datasources(config.langmodtrans_capgen_dataset)
capgen_size = datasources['train'].size
capgen_test = datasources['test']
del datasources

lib.create_dir(config.results_dir+'/langmodtrans')

for corpus in corpora:
    lib.create_dir(config.results_dir+'/langmodtrans/'+corpus)
    if not lib.file_exists(config.results_dir+'/langmodtrans/'+corpus+'/results.txt'):
        with open(config.results_dir+'/langmodtrans/'+corpus+'/results.txt', 'w', encoding='utf-8') as f:
            print(
                    'corpus',
                    'frozen_prefix',
                    'corpus_size_factor_exponent',
                    'run',
                    'corpus_size',
                    'langmod_vocab_size',
                    'langmod_num_params',
                    'langmod_mean_prob',
                    'langmod_median_prob',
                    'langmod_geomean_prob',
                    'langmod_mean_pplx',
                    'langmod_median_pplx',
                    'langmod_geomean_pplx',
                    'langmod_vocab_used',
                    'langmod_vocab_used_frac',
                    'langmod_min_freq_vocab_used',
                    'langmod_min_sent_len',
                    'langmod_mean_sent_len',
                    'langmod_max_sent_len',
                    'langmod_num_reused_sents',
                    'langmod_num_reused_sents_frac',
                    'langmod_reused_sents_WMD',
                    'langmod_num_reused_3grams',
                    'langmod_num_reused_4grams',
                    'langmod_num_reused_5grams',
                    'langmod_num_unique_sents',
                    'langmod_num_unique_sents_frac',
                    'langmod_num_types_nouns',
                    'langmod_num_types_adjectives',
                    'langmod_num_types_verbs',
                    'langmod_num_types_adverbs',
                    'langmod_num_types_unigrams',
                    'langmod_num_types_bigrams',
                    'langmod_num_types_trigrams',
                    'langmod_num_epochs',
                    'langmod_training_time',
                    'langmod_total_time',
                    'capgen_vocab_size',
                    'capgen_num_params',
                    'capgen_mean_prob',
                    'capgen_median_prob',
                    'capgen_geomean_prob',
                    'capgen_mean_pplx',
                    'capgen_median_pplx',
                    'capgen_geomean_pplx',
                    'capgen_vocab_used',
                    'capgen_vocab_used_frac',
                    'capgen_min_freq_vocab_used',
                    'capgen_min_sent_len',
                    'capgen_mean_sent_len',
                    'capgen_max_sent_len',
                    'capgen_num_reused_sents',
                    'capgen_num_reused_sents_frac',
                    'capgen_reused_sents_WMD',
                    'capgen_num_reused_3grams',
                    'capgen_num_reused_4grams',
                    'capgen_num_reused_5grams',
                    'capgen_num_unique_sents',
                    'capgen_num_unique_sents_frac',
                    'capgen_num_types_nouns',
                    'capgen_num_types_adjectives',
                    'capgen_num_types_verbs',
                    'capgen_num_types_adverbs',
                    'capgen_num_types_unigrams',
                    'capgen_num_types_bigrams',
                    'capgen_num_types_trigrams',
                    'capgen_BLEU_1',
                    'capgen_BLEU_2',
                    'capgen_BLEU_3',
                    'capgen_BLEU_4',
                    'capgen_METEOR',
                    'capgen_ROUGE_L',
                    'capgen_CIDEr',
                    'capgen_SPICE',
                    'capgen_WMD',
                    'capgen_R@1',
                    'capgen_R@5',
                    'capgen_R@10',
                    'capgen_median_rank',
                    'capgen_R@1_frac',
                    'capgen_R@5_frac',
                    'capgen_R@10_frac',
                    'capgen_median_rank_frac',
                    'capgen_num_epochs',
                    'capgen_training_time',
                    'capgen_total_time',
                    sep='\t', file=f
                )
    already_seen = set()
    with open(config.results_dir+'/langmodtrans/'+corpus+'/results.txt', 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n')[1:]:
            [
                frozen_prefix,
                corpus_size_factor_exponent,
                run,
            ] = line.split('\t')[1:4]
            
            next_exp = [
                corpus,
                frozen_prefix == 'True',
                float(corpus_size_factor_exponent),
                int(run),
            ]
            
            already_seen.add(tuple(next_exp))

    with open(config.hyperpar_dir+'/langmodtrans/'+corpus+'/1_best.txt', 'r', encoding='utf-8') as f:
        hyperpars = dict(line.split('\t') for line in f.read().strip().split('\n'))
    langmod_init_method             = hyperpars['init_method']
    langmod_max_init_weight         = float(hyperpars['max_init_weight'])
    langmod_embed_size              = int(hyperpars['embed_size'])
    langmod_rnn_size                = int(hyperpars['rnn_size'])
    langmod_post_image_size         = int(hyperpars['post_image_size']) if hyperpars['post_image_size'] != 'None' else None
    langmod_pre_output_size         = int(hyperpars['pre_output_size']) if hyperpars['pre_output_size'] != 'None' else None
    langmod_post_image_activation   = hyperpars['post_image_activation']
    langmod_rnn_type                = hyperpars['rnn_type']
    langmod_optimizer               = hyperpars['optimizer']
    langmod_learning_rate           = float(hyperpars['learning_rate'])
    langmod_normalize_image         = hyperpars['normalize_image'] == 'True'
    langmod_weights_reg_weight      = float(hyperpars['weights_reg_weight'])
    langmod_image_dropout_prob      = float(hyperpars['image_dropout_prob'])
    langmod_post_image_dropout_prob = float(hyperpars['post_image_dropout_prob'])
    langmod_embedding_dropout_prob  = float(hyperpars['embedding_dropout_prob'])
    langmod_rnn_dropout_prob        = float(hyperpars['rnn_dropout_prob'])
    langmod_max_gradient_norm       = float(hyperpars['max_gradient_norm'])
    langmod_minibatch_size          = int(hyperpars['minibatch_size'])
    langmod_beam_width              = int(hyperpars['beam_width'])
    
    with open(config.hyperpar_dir+'/langmodtrans/'+corpus+'/2_best.txt', 'r', encoding='utf-8') as f:
        hyperpars = dict(line.split('\t') for line in f.read().strip().split('\n'))
    capgen_init_method             = hyperpars['init_method']
    capgen_max_init_weight         = float(hyperpars['max_init_weight'])
    capgen_embed_size              = int(hyperpars['embed_size'])
    capgen_rnn_size                = int(hyperpars['rnn_size'])
    capgen_post_image_size         = int(hyperpars['post_image_size']) if hyperpars['post_image_size'] != 'None' else None
    capgen_pre_output_size         = int(hyperpars['pre_output_size']) if hyperpars['pre_output_size'] != 'None' else None
    capgen_post_image_activation   = hyperpars['post_image_activation']
    capgen_rnn_type                = hyperpars['rnn_type']
    capgen_optimizer               = hyperpars['optimizer']
    capgen_learning_rate           = float(hyperpars['learning_rate'])
    capgen_normalize_image         = hyperpars['normalize_image'] == 'True'
    capgen_weights_reg_weight      = float(hyperpars['weights_reg_weight'])
    capgen_image_dropout_prob      = float(hyperpars['image_dropout_prob'])
    capgen_post_image_dropout_prob = float(hyperpars['post_image_dropout_prob'])
    capgen_embedding_dropout_prob  = float(hyperpars['embedding_dropout_prob'])
    capgen_rnn_dropout_prob        = float(hyperpars['rnn_dropout_prob'])
    capgen_max_gradient_norm       = float(hyperpars['max_gradient_norm'])
    capgen_minibatch_size          = int(hyperpars['minibatch_size'])
    capgen_beam_width              = int(hyperpars['beam_width'])
    
    for frozen_prefix in [True, False]:
        for run in range(1, config.num_runs+1):
            for corpus_size_factor_exponent in (config.langmodtrans_corpus_size_factor_exponents if corpus != config.langmodtrans_capgen_dataset else config.langmodtrans_corpus_size_factor_minor_exponents):
                
                print('='*100)
                print(lib.formatted_clock())
                print(corpus, frozen_prefix, corpus_size_factor_exponent, run)
                print()
                
                if (corpus, frozen_prefix, corpus_size_factor_exponent, run) in already_seen:
                    print('Found ready')
                    print()
                    continue
                
                full_timer = lib.Timer()
                
                print('-'*100)
                print('Phase 1: langmod')
                print()
                
                dir_name = '{}_{}_{}'.format(frozen_prefix, corpus_size_factor_exponent, run)
                lib.create_dir(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name)
                
                corpus_size = round(10**corpus_size_factor_exponent * capgen_size)
        
                datasources = data.load_datasources(corpus)
                datasources['train'] = datasources['train'].without_images().shuffle(run).take(corpus_size)
                langmod_vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq)
                
                dataset = data.Dataset(
                        vocab            = langmod_vocab,
                        train_datasource = datasources['train'],
                        val_datasource   = datasources['val'],
                        test_datasource  = capgen_test,
                    )
                dataset.compile_sents()
                
                selected_test_sents = dataset.test.shuffle(run).take(one_per_group=True).tokenize_sents().compile_sents(langmod_vocab)
                selected_index_sents = selected_test_sents.index_sents
                
                with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_corpus_indexes.txt', 'w', encoding='utf-8') as f:
                    print(*dataset.train.individual_indexes, sep='\n', file=f)
                
                with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_selected_test.txt', 'w', encoding='utf-8') as f:
                    print(*selected_test_sents.individual_indexes, sep='\n', file=f)
                
                langmod_vocab.save_vocab(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_vocab.json')
                
                with model_neural_trad.TradNeuralModel(
                        vocab_size              = langmod_vocab.size,
                        init_method             = langmod_init_method,
                        max_init_weight         = langmod_max_init_weight,
                        embed_size              = langmod_embed_size,
                        rnn_size                = langmod_rnn_size,
                        post_image_size         = langmod_post_image_size,
                        pre_output_size         = langmod_pre_output_size,
                        post_image_activation   = langmod_post_image_activation,
                        rnn_type                = langmod_rnn_type,
                        architecture            = 'langmod',
                        optimizer               = langmod_optimizer,
                        learning_rate           = langmod_learning_rate,
                        normalize_image         = langmod_normalize_image,
                        weights_reg_weight      = langmod_weights_reg_weight,
                        image_dropout_prob      = langmod_image_dropout_prob,
                        post_image_dropout_prob = langmod_post_image_dropout_prob,
                        embedding_dropout_prob  = langmod_embedding_dropout_prob,
                        rnn_dropout_prob        = langmod_rnn_dropout_prob,
                        max_gradient_norm       = langmod_max_gradient_norm,
                        freeze_prefix_params    = False,
                    ) as model:
                    model.compile_model()
                    
                    model.init_params()
                    
                    train_timer = lib.Timer()
                    
                    print('Training...')
                    
                    langmod_fit_stats = model.fit(dataset, config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_model.hdf5', max_batch_size=config.val_batch_size, minibatch_size=langmod_minibatch_size, max_epochs=config.max_epochs, early_stop_patience=config.early_stop_patience, listener=_FitListener())
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_train.txt', 'w', encoding='utf-8') as f:
                        print('epoch', 'train logpplx', 'val logpplx', sep='\t', file=f)
                        for (epoch, train_logpplx, val_logpplx) in zip(range(langmod_fit_stats['num_epochs']+1), langmod_fit_stats['train_logpplxs'], langmod_fit_stats['val_logpplxs']):
                            print(epoch, train_logpplx, val_logpplx, sep='\t', file=f)
                    print()
                    
                    langmod_train_duration = train_timer.get_duration()
                    
                    print('Probability stats...')
                    print()
                    
                    (sents_logprobs, tokens_logprobs) = model.get_sents_logprobs(max_batch_size=config.val_batch_size, index_sents=dataset.test.index_sents)
                    langmod_prob_stats = evaluation.get_probability_stats(sents_logprobs, dataset.test.index_sents.lens)
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_probs.txt', 'w', encoding='utf-8') as f:
                        for logprobs in tokens_logprobs:
                            print(*logprobs, sep='\t', file=f)
                    
                    print('Generating sentences...')
                    
                    prog = lib.ProgressBar(capgen_test.num_groups, 5)
                    (index_sents, logprobs) = model.generate_sents_sample(max_batch_size=config.val_batch_size, images=[None]*capgen_test.num_groups, lower_bound_len=config.lower_bound_len, upper_bound_len=config.upper_bound_len, temperature=config.temperature, listener=lambda num_ready:prog.inc_value())
                    text_sents = index_sents.decompile_sents(langmod_vocab).sents
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_sents.txt', 'w', encoding='utf-8') as f:
                        for sent in text_sents:
                            print(*sent, sep=' ', file=f)
                    print()
                    print()
                    
                    print('Diversity stats...')
                    print()
                    
                    langmod_diversity_stats = evaluation.diversity_eval(dataset.train.text_sents.sents, dataset.test.get_text_sent_groups(), langmod_vocab, dataset.train.text_sents.get_vocab_freqs(), text_sents)
                    
                    langmod_num_params = model.get_num_params()
                
                langmod_full_duration = full_timer.get_duration()
                
                prefix_params = model_neural_trad.TradNeuralModel.get_saved_prefix_params(langmod_vocab, config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_model.hdf5')
                
                print('-'*100)
                print('Phase 2: capgen')
                print()
                
                full_timer = lib.Timer()
                
                datasources = data.load_datasources(config.langmodtrans_capgen_dataset)
                capgen_vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq).intersection(prefix_params.vocab)
                prefix_params = prefix_params.convert_to_new_vocabulary(capgen_vocab)
                
                dataset = data.Dataset(
                        vocab            = capgen_vocab,
                        train_datasource = datasources['train'],
                        val_datasource   = datasources['val'],
                        test_datasource  = datasources['test'],
                    )
                dataset.compile_sents()
                
                selected_test_sents = dataset.test.shuffle(run).take(one_per_group=True).tokenize_sents().compile_sents(capgen_vocab)
                selected_index_sents = selected_test_sents.index_sents
                
                with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_selected_test.txt', 'w', encoding='utf-8') as f:
                    print(*selected_test_sents.individual_indexes, sep='\n', file=f)
                
                capgen_vocab.save_vocab(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_vocab.json')
                
                with model_neural_trad.TradNeuralModel(
                        vocab_size              = capgen_vocab.size,
                        init_method             = capgen_init_method,
                        max_init_weight         = capgen_max_init_weight,
                        embed_size              = capgen_embed_size,
                        rnn_size                = capgen_rnn_size,
                        post_image_size         = capgen_post_image_size,
                        pre_output_size         = capgen_pre_output_size,
                        post_image_activation   = capgen_post_image_activation,
                        rnn_type                = capgen_rnn_type,
                        architecture            = 'merge',
                        optimizer               = capgen_optimizer,
                        learning_rate           = capgen_learning_rate,
                        normalize_image         = capgen_normalize_image,
                        weights_reg_weight      = capgen_weights_reg_weight,
                        image_dropout_prob      = capgen_image_dropout_prob,
                        post_image_dropout_prob = capgen_post_image_dropout_prob,
                        embedding_dropout_prob  = capgen_embedding_dropout_prob,
                        rnn_dropout_prob        = capgen_rnn_dropout_prob,
                        max_gradient_norm       = capgen_max_gradient_norm,
                        freeze_prefix_params    = frozen_prefix,
                    ) as model:
                    model.compile_model()
                    
                    model.init_params()
                    model.set_prefix_params(prefix_params)
                    
                    train_timer = lib.Timer()
                    
                    print('Training...')
                    
                    capgen_fit_stats = model.fit(dataset, config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_model.hdf5', max_batch_size=config.val_batch_size, minibatch_size=capgen_minibatch_size, max_epochs=config.max_epochs, early_stop_patience=config.early_stop_patience, listener=_FitListener())
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_train.txt', 'w', encoding='utf-8') as f:
                        print('epoch', 'train logpplx', 'val logpplx', sep='\t', file=f)
                        for (epoch, train_logpplx, val_logpplx) in zip(range(capgen_fit_stats['num_epochs']+1), capgen_fit_stats['train_logpplxs'], capgen_fit_stats['val_logpplxs']):
                            print(epoch, train_logpplx, val_logpplx, sep='\t', file=f)
                    print()
                    
                    capgen_train_duration = train_timer.get_duration()
                    
                    print('Probability stats...')
                    print()
                    
                    (sents_logprobs, tokens_logprobs) = model.get_sents_logprobs(max_batch_size=config.val_batch_size, index_sents=dataset.test.index_sents, images=dataset.test.images)
                    capgen_prob_stats = evaluation.get_probability_stats(sents_logprobs, dataset.test.index_sents.lens)
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_probs.txt', 'w', encoding='utf-8') as f:
                        for logprobs in tokens_logprobs:
                            print(*logprobs, sep='\t', file=f)
                    
                    print('Generating sentences...')
                    
                    prog = lib.ProgressBar(dataset.test.num_groups, 5)
                    (index_sents, logprobs) = model.generate_sents_beamsearch(max_batch_size=config.val_batch_size, images=dataset.test.get_images(), beam_width=capgen_beam_width, lower_bound_len=config.lower_bound_len, upper_bound_len=config.upper_bound_len, temperature=config.temperature, listener=lambda num_ready:prog.update_value(num_ready))
                    text_sents = index_sents.decompile_sents(capgen_vocab).sents
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_sents.txt', 'w', encoding='utf-8') as f:
                        for sent in text_sents:
                            print(*sent, sep=' ', file=f)
                    print()
                    print()
                    
                    print('Diversity stats...')
                    print()
                    
                    capgen_diversity_stats = evaluation.diversity_eval(dataset.train.text_sents.sents, dataset.test.get_text_sent_groups(), capgen_vocab, dataset.train.text_sents.get_vocab_freqs(), text_sents)
                    
                    print('Quality stats...')
                    print()
                    
                    capgen_generation_stats = evaluation.mscoco_eval(dataset.test.get_text_sent_groups(), text_sents)
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_wmd.txt', 'w', encoding='utf-8') as f:
                        print(*capgen_generation_stats['WMD_all'], sep='\n', file=f)
                    print()
                    
                    print('Generating image-caption probability matrix...')
                    
                    prog = lib.ProgressBar(dataset.test.num_groups**2, 5)
                    image_caption_logprobs_matrix = list()
                    for (sent_prefix, sent_len, sent_target) in zip(selected_index_sents.prefixes, selected_index_sents.lens, selected_index_sents.targets):
                        index_sents = data.IndexSents([sent_prefix]*selected_index_sents.size, [sent_len]*selected_index_sents.size, [sent_target]*selected_index_sents.size)
                        logprobs = model.get_sents_logprobs(max_batch_size=config.val_batch_size, index_sents=index_sents, images=dataset.test.get_images(), listener=lambda num_ready:prog.inc_value())[0]
                        image_caption_logprobs_matrix.append(logprobs)
                    print()
                    print()
                    
                    print('Retrieval stats...')
                    print()
                    
                    capgen_retrieval_stats = evaluation.retrieval_eval(image_caption_logprobs_matrix)
                    with open(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/2_rets.txt', 'w', encoding='utf-8') as f:
                        for row in image_caption_logprobs_matrix:
                            print(*row, sep='\t', file=f)
                    
                    capgen_num_params = model.get_num_params()
                        
                capgen_full_duration = full_timer.get_duration()
                
                print('Done!')
                
                with open(config.results_dir+'/langmodtrans/'+corpus+'/results.txt', 'a', encoding='utf-8') as f:
                    print(
                        corpus,
                        frozen_prefix,
                        corpus_size_factor_exponent,
                        run,
                        corpus_size,
                        langmod_vocab.size,
                        langmod_num_params,
                        langmod_prob_stats['mean_prob'],
                        langmod_prob_stats['median_prob'],
                        langmod_prob_stats['geomean_prob'],
                        langmod_prob_stats['mean_pplx'],
                        langmod_prob_stats['median_pplx'],
                        langmod_prob_stats['geomean_pplx'],
                        langmod_diversity_stats['vocab_used'],
                        langmod_diversity_stats['vocab_used_frac'],
                        langmod_diversity_stats['min_freq_vocab_used'],
                        langmod_diversity_stats['min_sent_len'],
                        langmod_diversity_stats['mean_sent_len'],
                        langmod_diversity_stats['max_sent_len'],
                        langmod_diversity_stats['num_reused_sents'],
                        langmod_diversity_stats['num_reused_sents_frac'],
                        langmod_diversity_stats['reused_sents_WMD'],
                        langmod_diversity_stats['num_reused_3grams'],
                        langmod_diversity_stats['num_reused_4grams'],
                        langmod_diversity_stats['num_reused_5grams'],
                        langmod_diversity_stats['num_unique_sents'],
                        langmod_diversity_stats['num_unique_sents_frac'],
                        langmod_diversity_stats['num_types_nouns'],
                        langmod_diversity_stats['num_types_adjectives'],
                        langmod_diversity_stats['num_types_verbs'],
                        langmod_diversity_stats['num_types_adverbs'],
                        langmod_diversity_stats['num_types_unigrams'],
                        langmod_diversity_stats['num_types_bigrams'],
                        langmod_diversity_stats['num_types_trigrams'],
                        langmod_fit_stats['num_epochs'],
                        langmod_train_duration,
                        langmod_full_duration,
                        capgen_vocab.size,
                        capgen_num_params,
                        capgen_prob_stats['mean_prob'],
                        capgen_prob_stats['median_prob'],
                        capgen_prob_stats['geomean_prob'],
                        capgen_prob_stats['mean_pplx'],
                        capgen_prob_stats['median_pplx'],
                        capgen_prob_stats['geomean_pplx'],
                        capgen_diversity_stats['vocab_used'],
                        capgen_diversity_stats['vocab_used_frac'],
                        capgen_diversity_stats['min_freq_vocab_used'],
                        capgen_diversity_stats['min_sent_len'],
                        capgen_diversity_stats['mean_sent_len'],
                        capgen_diversity_stats['max_sent_len'],
                        capgen_diversity_stats['num_reused_sents'],
                        capgen_diversity_stats['num_reused_sents_frac'],
                        capgen_diversity_stats['reused_sents_WMD'],
                        capgen_diversity_stats['num_reused_3grams'],
                        capgen_diversity_stats['num_reused_4grams'],
                        capgen_diversity_stats['num_reused_5grams'],
                        capgen_diversity_stats['num_unique_sents'],
                        capgen_diversity_stats['num_unique_sents_frac'],
                        capgen_diversity_stats['num_types_nouns'],
                        capgen_diversity_stats['num_types_adjectives'],
                        capgen_diversity_stats['num_types_verbs'],
                        capgen_diversity_stats['num_types_adverbs'],
                        capgen_diversity_stats['num_types_unigrams'],
                        capgen_diversity_stats['num_types_bigrams'],
                        capgen_diversity_stats['num_types_trigrams'],
                        capgen_generation_stats['Bleu_1'],
                        capgen_generation_stats['Bleu_2'],
                        capgen_generation_stats['Bleu_3'],
                        capgen_generation_stats['Bleu_4'],
                        capgen_generation_stats['METEOR'],
                        capgen_generation_stats['ROUGE_L'],
                        capgen_generation_stats['CIDEr'],
                        capgen_generation_stats['SPICE'],
                        capgen_generation_stats['WMD'],
                        capgen_retrieval_stats['R@1'],
                        capgen_retrieval_stats['R@5'],
                        capgen_retrieval_stats['R@10'],
                        capgen_retrieval_stats['median_rank'],
                        capgen_retrieval_stats['R@1_frac'],
                        capgen_retrieval_stats['R@5_frac'],
                        capgen_retrieval_stats['R@10_frac'],
                        capgen_retrieval_stats['median_rank_frac'],
                        capgen_fit_stats['num_epochs'],
                        capgen_train_duration,
                        capgen_full_duration,
                        sep='\t', file=f
                    )
                
                already_seen.add((corpus, frozen_prefix, corpus_size_factor_exponent, run))
                print()
                print(lib.format_duration(langmod_full_duration+capgen_full_duration))
                print()

print(lib.formatted_clock())