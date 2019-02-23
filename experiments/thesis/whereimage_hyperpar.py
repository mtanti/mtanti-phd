import skopt
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
def standardize_hyperpar(hp, architecture):
    new_hp = [
        (
            round(x.tolist(), 20) if type(x) is np.float64 else
            x.tolist() if type(x) is np.int64 else
            x
        ) for x in hp
    ]
    if architecture == 'init':
        new_hp[4] = new_hp[3]
    elif architecture == 'pre':
        new_hp[4] = new_hp[2]
    return new_hp
    
########################################################################################
def prepare_hyperpar_for_tell(hp, architecture):
    new_hp = list(hp)
    if architecture not in ['merge', 'par', 'merge-ext']:
        new_hp[4] = None
    return new_hp
    
########################################################################################
if len(sys.argv) == 1:
    architectures = 'merge,par,pre,init'.split(',') #can also add merge-ext for a two layered softmax in merge but this was found to work badly with the current hyperparameter search resources
else:
    architectures = sys.argv[1].split(',')

datasources = data.load_datasources('flickr8k')

vocab = datasources['train'].tokenize_sents().text_sents.get_vocab(config.min_token_freq)
dataset = data.Dataset(
        vocab            = vocab,
        train_datasource = datasources['train'],
        val_datasource   = datasources['val'],
        test_datasource  = data.load_datasources('mscoco')['val'].shuffle(0).take(datasources['test'].num_groups, whole_groups=True),
    )
dataset.compile_sents()

test_images = dataset.test.get_images()
test_sents  = dataset.test.get_text_sent_groups()

lib.create_dir(config.hyperpar_dir+'/whereimage')

for architecture in architectures:
    lib.create_dir(config.hyperpar_dir+'/whereimage/'+architecture)
    
    print('='*100)
    print(lib.formatted_clock())
    print(architecture)
    print()
    
    if lib.file_exists(config.hyperpar_dir+'/whereimage/'+architecture+'/best.txt'):
        print('Found ready')
        print()
        continue
    
    print(
            '#',
            'init_method',
            'max_init_weight',
            'embed_size',
            'rnn_size',
            'post_image_size',
            'pre_output_size',
            'post_image_activation',
            'rnn_type',
            'optimizer',
            'learning_rate',
            'normalize_image',
            'weights_reg_weight',
            'image_dropout_prob',
            'post_image_dropout_prob',
            'embedding_dropout_prob',
            'rnn_dropout_prob',
            'max_gradient_norm',
            'minibatch_size',
            'beam_width',
            'WMD',
            'duration',
            sep='\t'
        )
    
    if not lib.file_exists(config.hyperpar_dir+'/whereimage/'+architecture+'/search.txt'):
        with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search.txt', 'w', encoding='utf-8') as f:
            print(
                    '#',
                    'init_method',
                    'max_init_weight',
                    'embed_size',
                    'rnn_size',
                    'post_image_size',
                    'pre_output_size',
                    'post_image_activation',
                    'rnn_type',
                    'optimizer',
                    'learning_rate',
                    'normalize_image',
                    'weights_reg_weight',
                    'image_dropout_prob',
                    'post_image_dropout_prob',
                    'embedding_dropout_prob',
                    'rnn_dropout_prob',
                    'max_gradient_norm',
                    'minibatch_size',
                    'beam_width',
                    'WMD',
                    'duration',
                    sep='\t', file=f
                )

    if not lib.file_exists(config.hyperpar_dir+'/whereimage/'+architecture+'/search_errors.txt'):
        with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search_errors.txt', 'w', encoding='utf-8') as f:
            print(
                    '#',
                    'init_method',
                    'max_init_weight',
                    'embed_size',
                    'rnn_size',
                    'post_image_size',
                    'pre_output_size',
                    'post_image_activation',
                    'rnn_type',
                    'optimizer',
                    'learning_rate',
                    'normalize_image',
                    'weights_reg_weight',
                    'image_dropout_prob',
                    'post_image_dropout_prob',
                    'embedding_dropout_prob',
                    'rnn_dropout_prob',
                    'max_gradient_norm',
                    'minibatch_size',
                    'beam_width',
                    'error',
                    'duration',
                    sep='\t', file=f
                )

    def objective(hyperpar):
        [
            init_method,
            max_init_weight,
            embed_size,
            rnn_size,
            post_image_size,
            pre_output_size,
            post_image_activation,
            rnn_type,
            optimizer,
            learning_rate,
            normalize_image,
            weights_reg_weight,
            image_dropout_prob,
            post_image_dropout_prob,
            embedding_dropout_prob,
            rnn_dropout_prob,
            max_gradient_norm,
            minibatch_size,
            beam_width,
        ] = hyperpar
        with model_neural_trad.TradNeuralModel(
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
            ) as model:
            model.compile_model()
            
            result = list()
            for _ in range(config.hyperpar_num_runs):
                model.init_params()

                model.fit(dataset, config.hyperpar_dir+'/whereimage/'+architecture+'/model.hdf5', max_batch_size=config.val_batch_size, minibatch_size=minibatch_size, max_epochs=config.hyperpar_max_epochs, early_stop_patience=config.early_stop_patience)

                (index_sents, logprobs) = model.generate_sents_beamsearch(max_batch_size=config.val_batch_size, images=test_images, beam_width=beam_width, lower_bound_len=config.lower_bound_len, upper_bound_len=config.upper_bound_len, temperature=config.temperature)
                text_sents = index_sents.decompile_sents(vocab).sents
                
                wmd = evaluation.get_wmd_score(test_sents, text_sents)[0]
                
                result.append(wmd)
        
            return -np.mean(result)

    opt = skopt.Optimizer(
            [
                skopt.space.Categorical(config.hyperpar_space['init_method'], name='init_method'),
                skopt.space.Real(*config.hyperpar_space['max_init_weight'], 'log-uniform', name='max_init_weight'),
                skopt.space.Integer(*config.hyperpar_space['embed_size'], name='embed_size'),
                skopt.space.Integer(*config.hyperpar_space['rnn_size'], name='rnn_size'),
                skopt.space.Integer(*config.hyperpar_space['post_image_size'], name='post_image_size') if architecture in ['merge', 'merge-ext', 'par'] else skopt.space.Categorical([None], name='pre_output_size'),
                skopt.space.Integer(*config.hyperpar_space['pre_output_size'], name='pre_output_size') if architecture == 'merge-ext' else skopt.space.Categorical([None], name='pre_output_size'),
                skopt.space.Categorical(config.hyperpar_space['post_image_activation'], name='post_image_activation'),
                skopt.space.Categorical(config.hyperpar_space['rnn_type'], name='rnn_type'),
                skopt.space.Categorical(config.hyperpar_space['optimizer'], name='optimizer'),
                skopt.space.Real(*config.hyperpar_space['learning_rate'], 'log-uniform', name='learning_rate'),
                skopt.space.Categorical(config.hyperpar_space['normalize_image'], name='normalize_image'),
                skopt.space.Real(*config.hyperpar_space['weights_reg_weight'], 'log-uniform', name='weights_reg_weight'),
                skopt.space.Real(*config.hyperpar_space['image_dropout_prob'], 'uniform', name='image_dropout_prob'),
                skopt.space.Real(*config.hyperpar_space['post_image_dropout_prob'], 'uniform', name='post_image_dropout_prob'),
                skopt.space.Real(*config.hyperpar_space['embedding_dropout_prob'], 'uniform', name='embedding_dropout_prob'),
                skopt.space.Real(*config.hyperpar_space['rnn_dropout_prob'], 'uniform', name='rnn_dropout_prob'),
                skopt.space.Real(*config.hyperpar_space['max_gradient_norm'], 'log-uniform', name='max_gradient_norm'),
                skopt.space.Integer(*config.hyperpar_space['minibatch_size'], name='minibatch_size'),
                skopt.space.Integer(*config.hyperpar_space['beam_width'], name='beam_width'),
            ],
            n_initial_points=config.hyperpar_num_random_evals,
            base_estimator='RF',
            acq_func='EI',
            acq_optimizer='auto',
            random_state=0,
        )
        
    i = 0
    already_seen = set()
    best_hyperpar = None
    best_cost = None
    with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search.txt', 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n')[1:]:
            i += 1
            [
                entry_num,
                init_method,
                max_init_weight,
                embed_size,
                rnn_size,
                post_image_size,
                pre_output_size,
                post_image_activation,
                rnn_type,
                optimizer,
                learning_rate,
                normalize_image,
                weights_reg_weight,
                image_dropout_prob,
                post_image_dropout_prob,
                embedding_dropout_prob,
                rnn_dropout_prob,
                max_gradient_norm,
                minibatch_size,
                beam_width,
                cost,
                duration,
            ] = line.split('\t')
            
            next_hyperpar = [
                init_method,
                float(max_init_weight),
                int(embed_size),
                int(rnn_size),
                int(post_image_size),
                int(pre_output_size) if pre_output_size != 'None' else None,
                post_image_activation,
                rnn_type,
                optimizer,
                float(learning_rate),
                normalize_image == 'True',
                float(weights_reg_weight),
                float(image_dropout_prob),
                float(post_image_dropout_prob),
                float(embedding_dropout_prob),
                float(rnn_dropout_prob),
                float(max_gradient_norm),
                int(minibatch_size),
                int(beam_width),
            ]
            cost = -float(cost)
            duration = int(duration)
            
            if i < config.hyperpar_num_random_evals + config.hyperpar_num_evals:
                num_hyperpars = 1
                while standardize_hyperpar(opt.ask(num_hyperpars)[-1], architecture) != next_hyperpar: #If 'ask' does not give the saved value then there were multiple 'ask' calls in the previous session due to some <<SEEN>>, <<NAN>>, or <<EMPTY>> error
                    print('<<FOUND HYPERPARAMS THAT RESULTED IN ERRORS LAST TIME>>', sep='\t')
                    num_hyperpars += 1
                opt.tell(prepare_hyperpar_for_tell(next_hyperpar, architecture), cost)
            
            if best_cost is None or cost < best_cost:
                best_hyperpar = next_hyperpar
                best_cost = cost
            already_seen.add(tuple(next_hyperpar))
            
            print(i, *next_hyperpar, -cost, lib.format_duration(duration), '******' if cost == best_cost else '', sep='\t')
            
    for _ in range(i, config.hyperpar_num_random_evals + config.hyperpar_num_evals):
        i += 1
        num_hyperpars = 1
        while True:
            t = lib.Timer()
            next_hyperpar = standardize_hyperpar(opt.ask(num_hyperpars)[-1], architecture) #This allows us to get different hyperparameters every time the previous hyperparameters resulted in <<SEEN>>, <<NAN>>, or <<EMPTY>>
            num_hyperpars += 1
            
            print(i, *next_hyperpar, sep='\t', end='\t')
            
            if tuple(next_hyperpar) in already_seen:
                duration = t.get_duration()        
                print('<<SEEN>>', lib.format_duration(duration), sep='\t')
                with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search_errors.txt', 'a', encoding='utf-8') as f:
                    print(i, *next_hyperpar, 'seen', duration, sep='\t', file=f)
                continue
                
            try:
                cost = objective(next_hyperpar)
            except model_neural_trad.NotANumberError:
                duration = t.get_duration()        
                print('<<NAN>>', lib.format_duration(duration), sep='\t')
                with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search_errors.txt', 'a', encoding='utf-8') as f:
                    print(i, *next_hyperpar, 'nan', duration, sep='\t', file=f)
                continue
            except model_neural_trad.EmptyBeamError:
                duration = t.get_duration()        
                print('<<EMPTY>>', lib.format_duration(duration), sep='\t')
                with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search_errors.txt', 'a', encoding='utf-8') as f:
                    print(i, *next_hyperpar, 'empty', duration, sep='\t', file=f)
                continue
            
            break
        duration = t.get_duration()
        
        opt.tell(prepare_hyperpar_for_tell(next_hyperpar, architecture), cost)
        
        if best_cost is None or cost < best_cost:
            best_hyperpar = next_hyperpar
            best_cost = cost
        already_seen.add(tuple(next_hyperpar))
        
        print(-cost, lib.format_duration(duration), '******' if cost == best_cost else '', sep='\t')
        with open(config.hyperpar_dir+'/whereimage/'+architecture+'/search.txt', 'a', encoding='utf-8') as f:
            print(i, *next_hyperpar, -cost, duration, sep='\t', file=f)
        
    print('-'*100)
    print(lib.formatted_clock())
    print('best found:')
    print('', *best_hyperpar, -best_cost, sep='\t')
    print()
    with open(config.hyperpar_dir+'/whereimage/'+architecture+'/best.txt', 'w', encoding='utf-8') as f:
        print('WMD', -best_cost, sep='\t', file=f)
        print('init_method', best_hyperpar[0], sep='\t', file=f)
        print('max_init_weight', best_hyperpar[1], sep='\t', file=f)
        print('embed_size', best_hyperpar[2], sep='\t', file=f)
        print('rnn_size', best_hyperpar[3], sep='\t', file=f)
        print('post_image_size', best_hyperpar[4], sep='\t', file=f)
        print('pre_output_size', best_hyperpar[5], sep='\t', file=f)
        print('post_image_activation', best_hyperpar[6], sep='\t', file=f)
        print('rnn_type', best_hyperpar[7], sep='\t', file=f)
        print('optimizer', best_hyperpar[8], sep='\t', file=f)
        print('learning_rate', best_hyperpar[9], sep='\t', file=f)
        print('normalize_image', best_hyperpar[10], sep='\t', file=f)
        print('weights_reg_weight', best_hyperpar[11], sep='\t', file=f)
        print('image_dropout_prob', best_hyperpar[12], sep='\t', file=f)
        print('post_image_dropout_prob', best_hyperpar[13], sep='\t', file=f)
        print('embedding_dropout_prob', best_hyperpar[14], sep='\t', file=f)
        print('rnn_dropout_prob', best_hyperpar[15], sep='\t', file=f)
        print('max_gradient_norm', best_hyperpar[16], sep='\t', file=f)
        print('minibatch_size', best_hyperpar[17], sep='\t', file=f)
        print('beam_width', best_hyperpar[18], sep='\t', file=f)
    