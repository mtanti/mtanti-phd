import os
import numpy as np
import sys
from scipy.spatial import distance
from scipy import stats
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from framework import lib
from framework import model_neural_trad
from framework import evaluation
from framework import data
from framework import config

########################################################################################
if len(sys.argv) == 1:
    architectures = 'merge,par,pre,init'.split(',')
else:
    architectures = sys.argv[1].split(',')

lib.create_dir(config.results_dir+'/imageimportance')

for architecture in ['init', 'pre', 'par', 'merge']:
    lib.create_dir(config.results_dir+'/imageimportance/'+architecture)
    if not lib.file_exists(config.results_dir+'/imageimportance/'+architecture+'/results_logits.txt'):
        with open(config.results_dir+'/imageimportance/'+architecture+'/results_logits.txt', 'w', encoding='utf-8') as f:
            print(
                    'architecture',
                    'dataset',
                    'run',
                    'use_generated_sents',
                    'sent_len',
                    'token_index',
                    'min_logit',
                    'mean_logit',
                    'max_logit',
                    'total_time',
                    sep='\t', file=f
                )
    already_seen = set()
    with open(config.results_dir+'/imageimportance/'+architecture+'/results_logits.txt', 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n')[1:]:
            [
                architecture,
                dataset,
                run,
                use_generated_sents,
                sent_len,
                token_index
            ] = line.split('\t')[:6]
            
            next_exp = [
                architecture,
                dataset,
                int(run),
                use_generated_sents == 'True',
                int(sent_len),
                int(token_index)
            ]
            
            already_seen.add(tuple(next_exp))

    for dataset_name in ['flickr8k', 'flickr30k', 'mscoco']:
        datasources = data.load_datasources(dataset_name)
        vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq)
        datasources['test'].tokenize_sents()
        
        correct_images = datasources['test'].images
        with open(config.results_dir+'/imageimportance/foils_'+dataset_name+'.txt', 'r', encoding='utf-8') as f:
            foil_images = [ datasources['test'].images[int(i)] for i in f.read().strip().split('\n') ]
        
        for use_generated_sents in [ True, False ]:
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
            
            for run in range(1, config.num_runs+1):
                dir_name = '{}_{}_{}'.format(architecture, dataset_name, run)
                
                if use_generated_sents == True:
                    with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/sents.txt', 'r', encoding='utf-8') as f:
                        correct_sents = [ line.split(' ') for line in f.read().strip().split('\n') ]
                else:
                    correct_sents = datasources['test'].shuffle(seed=0).take(one_per_group=True).tokenize_sents().text_sents.sents
                    
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
                model.compile_model()
                model.set_params(model.load_params(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/model.hdf5'))
                
                with model:
                    for sent_len in sorted({len(sent) for sent in correct_sents}):
                        print('='*100)
                        print('Sentence length:', sent_len)
                        
                        filtered_correct_sents = list()
                        filtered_correct_images = list()
                        filtered_foil_images = list()
                        for (correct_sent, correct_image, foil_image) in zip(correct_sents, correct_images, foil_images):
                            if len(correct_sent) == sent_len:
                                filtered_correct_sents.append(correct_sent)
                                filtered_correct_images.append(correct_image)
                                filtered_foil_images.append(foil_image)
                        
                        print('Number of sentences:', len(filtered_correct_sents))
                        print()
                        if len(filtered_correct_sents) < 20:
                            print('Too little!')
                            print()
                            continue
                        
                        for token_index in range(sent_len+1):
                            print('-'*100)
                            print(lib.formatted_clock())
                            print(architecture, dataset_name, run, use_generated_sents, sent_len, token_index)
                            print()
                            
                            if (architecture, dataset_name, run, use_generated_sents, sent_len, token_index) in already_seen:
                                print('Found ready')
                                print()
                                continue
                            
                            full_timer = lib.Timer()
                            
                            filtered_sents_datasource = data.DataSource([ ' '.join(sent[:token_index]) for sent in filtered_correct_sents ])
                            filtered_sents_datasource.tokenize_sents().compile_sents(vocab)
                            
                            [
                                corr_post_images,
                                corr_multimodal_vectors,
                                corr_logits,
                                corr_predictions
                            ] = model.get_hidden_layers(max_batch_size=config.val_batch_size, prefixes=filtered_sents_datasource.index_sents.prefixes, prefixes_lens=filtered_sents_datasource.index_sents.lens, images=filtered_correct_images, temperature=1.0)
                            
                            min_logits = np.mean(np.min(corr_logits, axis=1))
                            mean_logits = np.mean(np.mean(corr_logits, axis=1))
                            max_logits = np.mean(np.max(corr_logits, axis=1))
                            
                            full_duration = full_timer.get_duration()
                            
                            with open(config.results_dir+'/imageimportance/'+architecture+'/results_logits.txt', 'a', encoding='utf-8') as f:
                                print(
                                    architecture,
                                    dataset_name,
                                    run,
                                    use_generated_sents,
                                    sent_len-1,
                                    token_index,
                                    min_logits,
                                    mean_logits,
                                    max_logits,
                                    full_duration,
                                    sep='\t', file=f
                                )
                            
                            already_seen.add((architecture, dataset_name, run, sent_len, token_index))
                            print(lib.format_duration(full_duration))
                            print()

print(lib.formatted_clock())