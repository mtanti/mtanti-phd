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
    if not lib.file_exists(config.results_dir+'/langmodtrans/'+corpus+'/val_pplx.txt'):
        with open(config.results_dir+'/langmodtrans/'+corpus+'/val_pplx.txt', 'w', encoding='utf-8') as f:
            print(
                    'corpus',
                    'frozen_prefix',
                    'corpus_size_factor_exponent',
                    'run',
                    'langmod_val_mean_prob',
                    'langmod_val_median_prob',
                    'langmod_val_geomean_prob',
                    'langmod_val_mean_pplx',
                    'langmod_val_median_pplx',
                    'langmod_val_geomean_pplx',
                    sep='\t', file=f
                )
    already_seen = set()
    with open(config.results_dir+'/langmodtrans/'+corpus+'/val_pplx.txt', 'r', encoding='utf-8') as f:
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
                
                dir_name = '{}_{}_{}'.format(frozen_prefix, corpus_size_factor_exponent, run)
                lib.create_dir(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name)
                
                corpus_size = round(10**corpus_size_factor_exponent * capgen_size)
                
                full_timer = lib.Timer()
        
                datasources = data.load_datasources(corpus)
                datasources['train'] = datasources['train'].without_images().shuffle(run).take(corpus_size)
                langmod_vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq)
                
                capgen_vocab = capgen_test.tokenize_sents().text_sents.get_vocab(config.min_token_freq).intersection(langmod_vocab)
                capgen_full_vocab = capgen_test.tokenize_sents().text_sents.get_vocab()
                
                capgen_num_out_of_vocab_tokens = capgen_full_vocab.size - capgen_vocab.size
                
                dataset = data.Dataset(
                        vocab            = langmod_vocab,
                        train_datasource = datasources['train'],
                        val_datasource   = datasources['val'],
                    )
                dataset.compile_sents()
                
                capgen_num_unknowns_per_sent = np.sum(dataset.val.index_sents.targets == data.Vocab.UNKNOWN_INDEX, axis=1).tolist()
                
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
                    model.set_params(model.load_params(config.results_dir+'/langmodtrans/'+corpus+'/'+dir_name+'/1_model.hdf5'))
                    
                    (sents_logprobs, tokens_logprobs) = model.get_sents_logprobs(max_batch_size=config.val_batch_size, index_sents=dataset.val.index_sents)
                    val_langmod_prob_stats = evaluation.get_probability_stats(sents_logprobs, dataset.val.index_sents.lens)
                    langmod_prob_stats = evaluation.get_probability_stats(sents_logprobs, dataset.val.index_sents.lens, capgen_num_unknowns_per_sent, capgen_num_out_of_vocab_tokens)
                    
                print('Done!')
                print(lib.format_duration(full_timer.get_duration()))
                
                with open(config.results_dir+'/langmodtrans/'+corpus+'/val_pplx.txt', 'a', encoding='utf-8') as f:
                    print(
                        corpus,
                        frozen_prefix,
                        corpus_size_factor_exponent,
                        run,
                        langmod_prob_stats['mean_prob'],
                        langmod_prob_stats['median_prob'],
                        langmod_prob_stats['geomean_prob'],
                        langmod_prob_stats['mean_pplx'],
                        langmod_prob_stats['median_pplx'],
                        langmod_prob_stats['geomean_pplx'],
                        sep='\t', file=f
                    )
                
                already_seen.add((corpus, frozen_prefix, corpus_size_factor_exponent, run))
                print()
                print()

print(lib.formatted_clock())