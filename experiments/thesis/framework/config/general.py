debug = False

results_dir = 'results' if not debug else '_results_test'
hyperpar_dir = 'hyperparams' if not debug else '_hyperparams_test'
dataset_dir = 'data' if not debug else '_data_test'

langmodtrans_capgen_dataset = 'flickr8k'
langmodtrans_corpus_size_factor_exponents = [-1.0, -0.5, 0.0, 0.5, 1.0] if not debug else [-0.5, 0.0, 0.5]
langmodtrans_corpus_size_factor_minor_exponents = [-1.0, -0.5, 0.0] if not debug else [-0.5, 0.0]

partialtraining_max_max_epochs = 20
partialtraining_max_attempts = 5

min_token_freq = 5
google_val_files_used = 2
google_max_sent_len = 50

max_epochs = 100 if not debug else 2
num_runs = 5 if not debug else 2
early_stop_patience = 1

lower_bound_len = 5
upper_bound_len = 20
temperature = 1.0

hyperpar_num_runs = 2
hyperpar_max_epochs = 10 if not debug else 2
hyperpar_num_random_evals = 32 if not debug else 2
hyperpar_num_evals = 64 if not debug else 5
hyperpar_space = dict(
    init_method             = ['normal', 'xavier_normal'],
    max_init_weight         = (1e-5, 1.0),
    embed_size              = (64, 512),
    rnn_size                = (64, 512),
    post_image_size         = (64, 512),
    pre_output_size         = (64, 512),
    post_image_activation   = ['none', 'relu'],
    rnn_type                = ['gru'],
    optimizer               = ['adam', 'rmsprop', 'adadelta'],
    learning_rate           = (1e-5, 1.0),
    normalize_image         = [False, True],
    weights_reg_weight      = (1e-10, 0.1),
    image_dropout_prob      = (0.0, 0.5),
    post_image_dropout_prob = (0.0, 0.5),
    embedding_dropout_prob  = (0.0, 0.5),
    rnn_dropout_prob        = (0.0, 0.5),
    max_gradient_norm       = (1.0, 1000.0),
    minibatch_size          = (10, 300),
    beam_width              = (1, 5),
)