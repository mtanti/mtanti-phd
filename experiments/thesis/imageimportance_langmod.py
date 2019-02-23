import os
import numpy as np
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from framework import lib
from framework import model_neural_trad
from framework import evaluation
from framework import data
from framework import config

########################################################################################
lib.create_dir(config.results_dir+'/imageimportance')

architecture = 'langmod'
lib.create_dir(config.results_dir+'/imageimportance/'+architecture)
if not lib.file_exists(config.results_dir+'/imageimportance/'+architecture+'/results_langmod.txt'):
    with open(config.results_dir+'/imageimportance/'+architecture+'/results_langmod.txt', 'w', encoding='utf-8') as f:
        print(
                'dataset',
                'run',
                'sent_len',
                'token_index',
                'gradient_wrt_prefix_next',
                'gradient_wrt_prefix_max',
                'gradient_wrt_prevtoken_next',
                'gradient_wrt_prevtoken_max',
                'gradient_wrt_firsttoken_next',
                'gradient_wrt_firsttoken_max',
                'gradient_wrt_multimodalvec_next',
                'gradient_wrt_multimodalvec_max',
                'total_time',
                sep='\t', file=f
            )
already_seen = set()
with open(config.results_dir+'/imageimportance/'+architecture+'/results_langmod.txt', 'r', encoding='utf-8') as f:
    for line in f.read().strip().split('\n')[1:]:
        [
            dataset,
            run,
            sent_len,
            token_index
        ] = line.split('\t')[:6]
        
        next_exp = [
            dataset,
            int(run),
            int(sent_len),
            int(token_index)
        ]
        
        already_seen.add(tuple(next_exp))

for dataset_name in ['mscoco']:
    datasources = data.load_datasources(dataset_name)
    datasources['test'].tokenize_sents()
    
    for run in range(1, config.num_runs+1):
        dir_name = '{}_{}_{}'.format(architecture, dataset_name, run)
        lm_dir_name = '{}_{}_{}'.format(True, 1.0, run)
        
        langmod_vocab = data.Vocab.load_vocab(config.results_dir+'/langmodtrans/'+dataset_name+'/'+lm_dir_name+'/1_vocab.json')
    
        with open(config.hyperpar_dir+'/langmodtrans/'+dataset_name+'/1_best.txt', 'r', encoding='utf-8') as f:
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
        
        correct_sents = datasources['test'].tokenize_sents().text_sents.sents
            
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
            model.set_params(model.load_params(config.results_dir+'/langmodtrans/'+dataset_name+'/'+lm_dir_name+'/1_model.hdf5'))
            
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
                
                for token_index in range(sent_len+1):
                    print('-'*100)
                    print(lib.formatted_clock())
                    print(dataset_name, run, sent_len, token_index)
                    print()
                    
                    if (dataset_name, run, sent_len, token_index) in already_seen:
                        print('Found ready')
                        print()
                        continue
                    
                    full_timer = lib.Timer()
                    
                    filtered_sents_datasource = data.DataSource([ ' '.join(sent[:token_index]) for sent in filtered_correct_sents ])
                    filtered_sents_datasource.tokenize_sents().compile_sents(langmod_vocab)
                    
                    #sensitivity analysis
                    [
                        grad_next_wrt_prefix, grad_max_wrt_prefix,
                        grad_next_wrt_prevtoken, grad_max_wrt_prevtoken,
                        grad_next_wrt_firsttoken, grad_max_wrt_firsttoken,
                        grad_next_wrt_multimodalvec, grad_max_wrt_multimodalvec,
                    ] = model.get_sensitivity(images=None, prefixes=filtered_sents_datasource.index_sents.prefixes, prefixes_lens=filtered_sents_datasource.index_sents.lens, targets=filtered_sents_datasource.index_sents.targets, temperature=1.0)
                    
                    mean_grad_next_wrt_prefix = np.mean(np.mean(grad_next_wrt_prefix, axis=1))
                    mean_grad_max_wrt_prefix = np.mean(np.mean(grad_max_wrt_prefix, axis=1))
                    mean_grad_next_wrt_prevtoken = np.mean(np.mean(grad_next_wrt_prevtoken, axis=1))
                    mean_grad_max_wrt_prevtoken = np.mean(np.mean(grad_max_wrt_prevtoken, axis=1))
                    mean_grad_next_wrt_firsttoken = np.mean(np.mean(grad_next_wrt_firsttoken, axis=1))
                    mean_grad_max_wrt_firsttoken = np.mean(np.mean(grad_max_wrt_firsttoken, axis=1))
                    mean_grad_next_wrt_multimodalvec = np.mean(np.mean(grad_next_wrt_multimodalvec, axis=1))
                    mean_grad_max_wrt_multimodalvec = np.mean(np.mean(grad_max_wrt_multimodalvec, axis=1))
                    
                    full_duration = full_timer.get_duration()
                    
                    with open(config.results_dir+'/imageimportance/'+architecture+'/results_langmod.txt', 'a', encoding='utf-8') as f:
                        print(
                            dataset_name,
                            run,
                            sent_len-1,
                            token_index,
                            mean_grad_next_wrt_prefix,
                            mean_grad_max_wrt_prefix,
                            mean_grad_next_wrt_prevtoken,
                            mean_grad_max_wrt_prevtoken,
                            mean_grad_next_wrt_firsttoken,
                            mean_grad_max_wrt_firsttoken,
                            mean_grad_next_wrt_multimodalvec,
                            mean_grad_max_wrt_multimodalvec,
                            full_duration,
                            sep='\t', file=f
                        )
                    
                    already_seen.add((architecture, dataset_name, run, sent_len, token_index))
                    print(lib.format_duration(full_duration))
                    print()

print(lib.formatted_clock())