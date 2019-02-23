import numpy as np
import tensorflow as tf
import h5py

from . import lib
from . import model_neural
from . import evaluation
from .model_neural import FitListener
from .model_neural import NotANumberError
from .model_neural import EmptyBeamError

########################################################################################
class PrefixParams(object):

    ############################################
    def __init__(self, vocab, params):
        self.vocab = vocab
        self.params = params
        
    ############################################
    def convert_to_new_vocabulary(self, new_vocab):
        if not new_vocab.vocab_set <= self.vocab.vocab_set:
            raise ValueError('New vocabulary must be a subset of previous vocabulary (use intersection of two vocabularies)')
        
        new_params = dict(self.params)
        
        old_embedding_matrix = self.params['nn/prefix/embedding/embedding_matrix:0']
        token_indexes = [ self.vocab.token_to_index[token] for token in new_vocab.vocab_list ]
        new_embedding_matrix = old_embedding_matrix[token_indexes]
        
        new_params['nn/prefix/embedding/embedding_matrix:0'] = new_embedding_matrix
        
        return PrefixParams(new_vocab, new_params)
    
        
########################################################################################
class TradNeuralModel(model_neural.NeuralModel):

    ############################################
    def __init__(self, vocab_size, init_method, max_init_weight, embed_size, rnn_size, post_image_size, pre_output_size, post_image_activation, rnn_type, architecture, optimizer, learning_rate, normalize_image, weights_reg_weight, image_dropout_prob, post_image_dropout_prob, embedding_dropout_prob, rnn_dropout_prob, max_gradient_norm, freeze_prefix_params):
        '''
        init_method: normal, xavier_normal
        post_image_activation: none, relu
        rnn_type: srnn, gru
        architecture: init, pre, par, merge, merge-ext, langmod
        optimizer: adam, rmsprop, adadelta
        '''
        super(TradNeuralModel, self).__init__()
        
        if init_method not in 'normal, xavier_normal'.split(', '):
            raise ValueError('Invalid init_method ({})'.format(init_method))
        if post_image_activation not in 'none, relu'.split(', '):
            raise ValueError('Invalid post_image_activation ({})'.format(post_image_activation))
        if rnn_type not in 'srnn, gru'.split(', '):
            raise ValueError('Invalid rnn_type ({})'.format(rnn_type))
        if architecture not in 'init, pre, par, merge, merge-ext, langmod'.split(', '):
            raise ValueError('Invalid architecture ({})'.format(architecture))
        if optimizer not in 'adam, rmsprop, adadelta'.split(', '):
            raise ValueError('Invalid optimizer ({})'.format(optimizer))
        if architecture != 'langmod' and post_image_size is None:
            raise ValueError('Post-image size must be defined for architectures other than langmod')
        elif architecture == 'langmod' and post_image_size is not None:
            raise ValueError('Pre-output size must be None for architecture langmod')
        if architecture != 'merge-ext' and pre_output_size is not None:
            raise ValueError('Pre-output size must be None for architectures other than merge-ext')
        elif architecture == 'merge-ext' and pre_output_size is None:
            raise ValueError('Pre-output size must be defined for architecture merge-ext')
        if architecture == 'init' and rnn_size != post_image_size:
            raise ValueError('Init multimodal method requires that rnn size and post image size be equal ({} != {})'.format(rnn_size, post_image_size))
        if architecture == 'pre' and embed_size != post_image_size:
            raise ValueError('Pre multimodal method requires that embed size and post image size be equal ({} != {})'.format(embed_size, post_image_size))
        
        self.vocab_size              = vocab_size
        self.init_method             = init_method
        self.max_init_weight         = max_init_weight
        self.embed_size              = embed_size
        self.rnn_size                = rnn_size
        self.image_size              = 4096
        self.post_image_size         = post_image_size
        self.pre_output_size         = pre_output_size
        self.post_image_activation   = post_image_activation
        self.rnn_type                = rnn_type
        self.architecture            = architecture
        self.optimizer               = optimizer
        self.learning_rate           = learning_rate
        self.normalize_image         = normalize_image
        self.weights_reg_weight      = weights_reg_weight
        self.image_dropout_prob      = image_dropout_prob
        self.post_image_dropout_prob = post_image_dropout_prob
        self.embedding_dropout_prob  = embedding_dropout_prob
        self.rnn_dropout_prob        = rnn_dropout_prob
        self.max_gradient_norm       = max_gradient_norm
        self.freeze_prefix_params    = freeze_prefix_params
        
        self.prefixes           = None
        self.prefixes_lens      = None
        self.images             = None
        self.new_tokens         = None
        self.curr_states        = None
        self.temperature        = None
        self.dropout            = None
        self.init_states        = None
        self.post_images        = None
        self.embedded_seq       = None
        self.multimodal_vectors = None
        self.new_states         = None
        self.targets            = None
        self.logits             = None
        self.predictions        = None
        self.target_predictions = None
        self.stream_predictions = None
        self.loss               = None
        self.is_loss_nan        = None
        self.grad_nextpred_wrt_image     = None
        self.grad_maxpred_wrt_image      = None
        self.grad_nextpred_wrt_prefix    = None
        self.grad_maxpred_wrt_prefix     = None
        self.grad_nextpred_wrt_prevtoken = None
        self.grad_maxpred_wrt_prevtoken  = None
        self.last_multimodal_vectors     = None
        self.last_logits        = None
        self.last_predictions   = None
        self.train_step         = None
        self.num_params         = None
    
    ############################################
    def get_num_params(self):
        return self.num_params
        
    ############################################
    def compile_model(self):
        if self.init_method == 'xavier_normal':
            xavier = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
        def init(shape, dtype=None, partition_info=None):
            if len(shape) == 1:
                return tf.zeros(shape, dtype=dtype)
            else:
                if self.init_method == 'normal':
                    return tf.clip_by_value(tf.random_normal(shape, dtype=dtype), -self.max_init_weight, self.max_init_weight)
                elif self.init_method == 'xavier_normal':
                    return tf.clip_by_value(xavier(shape, dtype, partition_info), -self.max_init_weight, self.max_init_weight)
                    
        with self.graph.as_default():
            self.batch_size    = batch_size    = tf.placeholder(tf.int32,   [],                        'batch_size')
            self.prefixes      = prefixes      = tf.placeholder(tf.int32,   [ None, None ],            'prefixes')
            self.prefixes_lens = prefixes_lens = tf.placeholder(tf.int32,   [ None ],                  'prefixes_lens')
            self.images        = images        = tf.placeholder(tf.float32, [ None, self.image_size ], 'images')
            self.dropout       = dropout       = tf.placeholder(tf.bool,    [],                        'dropout')
            self.temperature   = temperature   = tf.placeholder(tf.float32, [],                        'temperature')
            self.targets       = targets       = tf.placeholder(tf.int32,   [ None, None ],            'targets')
            self.new_tokens    = new_tokens    = tf.placeholder(tf.int32,   [ None ],                  'new_tokens')
            self.curr_states   = curr_states   = tf.placeholder(tf.float32, [ None, self.rnn_size ],   'curr_states')
            
            image_dropout_keep_prob      = tf.cond(dropout, lambda:tf.constant(1.0-self.image_dropout_prob, tf.float32), lambda:tf.constant(1.0, tf.float32))
            post_image_dropout_keep_prob = tf.cond(dropout, lambda:tf.constant(1.0-self.post_image_dropout_prob, tf.float32), lambda:tf.constant(1.0, tf.float32))
            embedding_dropout_keep_prob  = tf.cond(dropout, lambda:tf.constant(1.0-self.embedding_dropout_prob, tf.float32), lambda:tf.constant(1.0, tf.float32))
            rnn_dropout_keep_prob        = tf.cond(dropout, lambda:tf.constant(1.0-self.rnn_dropout_prob, tf.float32), lambda:tf.constant(1.0, tf.float32))
            
            num_steps  = tf.shape(prefixes)[1]
            token_mask = tf.sequence_mask(prefixes_lens, num_steps, dtype=tf.float32)
            
            with tf.variable_scope('nn', initializer=init):
                if self.architecture != 'langmod':
                    with tf.variable_scope('image'):
                        W = tf.get_variable('W', [ self.image_size, self.post_image_size ], tf.float32)
                        b = tf.get_variable('b', [ self.post_image_size ], tf.float32)
                        images = tf.nn.dropout(images, image_dropout_keep_prob)
                        post_images = tf.matmul(images, W) + b
                        if self.post_image_activation == 'relu':
                            post_images = tf.nn.relu(post_images)
                        self.post_images = post_images
                        if self.architecture != 'init':
                            post_images = tf.expand_dims(post_images, 1)
                            if self.architecture != 'pre':
                                post_images = tf.tile(post_images, [ 1, num_steps, 1 ])
                        post_images = tf.nn.dropout(post_images, post_image_dropout_keep_prob)

                with tf.variable_scope('prefix'):
                    with tf.variable_scope('embedding'):
                        embedding_matrix = tf.get_variable('embedding_matrix', [ self.vocab_size, self.embed_size ], tf.float32)
                        
                        embedded_seq = tf.nn.embedding_lookup(embedding_matrix, prefixes)
                        self.embedded_seq = embedded_seq
                        embedded_seq = tf.nn.dropout(embedded_seq, embedding_dropout_keep_prob)
                        if self.architecture == 'pre':
                            embedded_seq = tf.concat([ post_images, embedded_seq ], axis=1)
                        elif self.architecture == 'par':
                            embedded_seq = tf.concat([ post_images, embedded_seq ], axis=2)
                        if self.architecture == 'pre':
                            embedded_seq_lens = prefixes_lens + 1 #Add 1 to all prefixes_lens if using pre-inject since image is included as a token
                        else:
                            embedded_seq_lens = prefixes_lens

                    with tf.variable_scope('rnn'):
                        if self.rnn_type == 'srnn':
                            cell = tf.contrib.rnn.BasicRNNCell(self.rnn_size)
                        elif self.rnn_type == 'gru':
                            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
                            
                        if self.architecture == 'init':
                            self.init_states = init_states = post_images
                        else:
                            init_state = tf.get_variable('init', [ self.rnn_size ], tf.float32)
                            self.init_states = init_states = tf.tile(tf.reshape(init_state, [ 1, self.rnn_size ]), [ batch_size, 1 ])
                            if self.architecture == 'pre':
                                self.init_states = cell(self.post_images, init_states)[0] #This simplifies the use of pre-inject in streaming generation.
                        
                    prefix_vectors = tf.nn.dynamic_rnn(cell, embedded_seq, sequence_length=embedded_seq_lens, initial_state=init_states)[0]
                    prefix_vectors = tf.nn.dropout(prefix_vectors, rnn_dropout_keep_prob)
                    
                    if self.architecture in [ 'merge', 'merge-ext' ]:
                        prefix_vectors = tf.concat([ post_images, prefix_vectors ], axis=2)
                    elif self.architecture == 'pre':
                        prefix_vectors = prefix_vectors[:,1:,:] #drop the prefix vector resulting from the image
                    
                    if self.architecture in [ 'merge', 'merge-ext' ]:
                        prefix_vector_size = self.post_image_size + self.rnn_size
                    else:
                        prefix_vector_size = self.rnn_size
                    prefix_vectors_2d = tf.reshape(prefix_vectors, [ batch_size*num_steps, prefix_vector_size ])
                    self.multimodal_vectors = prefix_vectors
                    
                if self.architecture == 'merge-ext':
                    with tf.variable_scope('pre_out'):
                        pre_out_W = tf.get_variable('W', [ prefix_vector_size, self.pre_output_size ], tf.float32)
                        pre_out_b = tf.get_variable('b', [ self.pre_output_size ],                     tf.float32)
                        pre_out = tf.nn.relu(tf.matmul(prefix_vectors_2d, pre_out_W) + pre_out_b)
                    
                    with tf.variable_scope('out'):
                        out_W = tf.get_variable('W', [ self.pre_output_size, self.vocab_size ], tf.float32)
                        out_b = tf.get_variable('b', [ self.vocab_size ],                       tf.float32)
                        logits = tf.matmul(pre_out, out_W) + out_b
                        logits = tf.reshape(logits, [ batch_size, num_steps, self.vocab_size ])
                else:
                    with tf.variable_scope('out'):
                        out_W = tf.get_variable('W', [ prefix_vector_size, self.vocab_size ], tf.float32)
                        out_b = tf.get_variable('b', [ self.vocab_size ],                     tf.float32)
                        logits = tf.matmul(prefix_vectors_2d, out_W) + out_b
                        self.logits = logits = tf.reshape(logits, [ batch_size, num_steps, self.vocab_size ])
                        
                self.predictions = tf.nn.softmax(logits/temperature)
                self.target_predictions = tf.reshape(
                    tf.gather_nd(
                        tf.reshape(self.predictions, [ batch_size*num_steps, self.vocab_size ]),
                        tf.stack([
                            tf.range(batch_size*num_steps),
                            tf.reshape(targets, [batch_size*num_steps])
                        ], axis=1)
                    ),
                    [ batch_size, num_steps ]
                )
                
                with tf.variable_scope('loss'):
                    weights_reg = tf.nn.l2_loss(tf.concat([ tf.reshape(v, [-1]) for v in tf.trainable_variables() if len(v.shape) == 2 ], axis=0))
                    cross_ent = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)*token_mask)/tf.cast(tf.reduce_sum(prefixes_lens), tf.float32)
                    self.loss = cross_ent + self.weights_reg_weight*weights_reg
                    self.is_loss_nan = tf.is_nan(self.loss)

            #Streaming generator code
            with tf.variable_scope('stream'):
                embedded_tokens = tf.nn.embedding_lookup(embedding_matrix, new_tokens)
                if self.architecture == 'par':
                    embedded_tokens = tf.concat([ self.post_images, embedded_tokens ], axis=1)
                
                self.new_states = new_states = cell(embedded_tokens, curr_states)[0]
                if self.architecture in [ 'merge', 'merge-ext' ]:
                    new_states = tf.concat([ self.post_images, new_states ], axis=1)
                
                if self.architecture == 'merge-ext':
                    pre_out = tf.nn.relu(tf.matmul(new_states, pre_out_W) + pre_out_b)
                    stream_logits = tf.matmul(pre_out, out_W) + out_b
                else:
                    stream_logits = tf.matmul(new_states, out_W) + out_b
                    
                self.stream_predictions = tf.nn.softmax(stream_logits/temperature)
            
            if self.freeze_prefix_params:
                trainable_vars = [ v for v in tf.trainable_variables() if not v.name.startswith('nn/prefix') ]
            else:
                trainable_vars = tf.trainable_variables()
            
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            
            grads = tf.gradients(self.loss, trainable_vars)
            (grads, _) = tf.clip_by_global_norm(grads, self.max_gradient_norm)
            self.train_step = optimizer.apply_gradients(zip(grads, trainable_vars))
            
            nextpred = self.predictions[0, -1, targets[0, -1]]
            maxpred = self.predictions[0, -1, tf.argmax(self.predictions[0, -1], output_type=tf.int32)]
            if self.architecture != 'langmod':
                self.grad_nextpred_wrt_image = tf.abs(tf.reshape(tf.gradients(nextpred, self.images)[0][0, :], [1, -1]))
                self.grad_maxpred_wrt_image = tf.abs(tf.reshape(tf.gradients(maxpred, self.images)[0][0, :], [1, -1]))
            self.grad_nextpred_wrt_prefix = tf.abs(tf.reshape(tf.gradients(nextpred, self.embedded_seq)[0][0, :, :], [1, -1]))
            self.grad_maxpred_wrt_prefix = tf.abs(tf.reshape(tf.gradients(maxpred, self.embedded_seq)[0][0, :, :], [1, -1]))
            self.grad_nextpred_wrt_prevtoken = tf.abs(tf.reshape(tf.gradients(nextpred, self.embedded_seq)[0][0, -1, :], [1, -1]))
            self.grad_maxpred_wrt_prevtoken = tf.abs(tf.reshape(tf.gradients(maxpred, self.embedded_seq)[0][0, -1, :], [1, -1]))
            self.grad_nextpred_wrt_firsttoken = tf.abs(tf.reshape(tf.gradients(nextpred, self.embedded_seq)[0][0, 0, :], [1, -1]))
            self.grad_maxpred_wrt_firsttoken = tf.abs(tf.reshape(tf.gradients(maxpred, self.embedded_seq)[0][0, 0, :], [1, -1]))
            if self.architecture != 'langmod':
                self.grad_nextpred_wrt_postimage = tf.abs(tf.reshape(tf.gradients(nextpred, self.post_images)[0][0, :], [1, -1]))
                self.grad_maxpred_wrt_postimage = tf.abs(tf.reshape(tf.gradients(maxpred, self.post_images)[0][0, :], [1, -1]))
            self.grad_nextpred_wrt_multimodalvec = tf.abs(tf.reshape(tf.gradients(nextpred, self.multimodal_vectors)[0][0, -1, :], [1, -1]))
            self.grad_maxpred_wrt_multimodalvec = tf.abs(tf.reshape(tf.gradients(maxpred, self.multimodal_vectors)[0][0, -1, :], [1, -1]))
            
            self.last_multimodal_vectors = self.multimodal_vectors[:, -1, :]
            self.last_logits = self.logits[:, -1, :]
            self.last_predictions = self.predictions[:, -1, :]
            
            self.initializer = tf.global_variables_initializer()
            
            for v in tf.trainable_variables():
                p = tf.placeholder(v.dtype, v.shape, v.name.split(':')[0]+'_setter')
                self.param_setters[v.name] = (tf.assign(v, p), p)
            
            self.num_params = 0
            for v in tf.trainable_variables():
                self.num_params += np.prod(v.shape.as_list())
            
            self.graph.finalize()
    
    
    ############################################
    def init_params(self):
        self.session.run(self.initializer)
    
    
    ############################################
    def fit(self, dataset, param_save_dir, max_batch_size, minibatch_size, max_epochs, last_epoch_max_minibatches=None, early_stop_patience=None, listener=FitListener()):
        listener.fit_started(self)
        
        best_val_logpplx = np.inf
        val_logpplxs = list()
        train_logpplxs = list()
        num_times_val_logpplx_worse = 0
        for epoch in range(0, max_epochs+1):
            listener.epoch_started(self, epoch)
            
            #Training
            if epoch > 0:
                self.train_params(
                            max_batch_size = minibatch_size,
                            prefixes       = dataset.train.index_sents.prefixes,
                            prefixes_lens  = dataset.train.index_sents.lens,
                            images         = dataset.train.images,
                            targets        = dataset.train.index_sents.targets,
                            max_batches    = last_epoch_max_minibatches if epoch == max_epochs else None,
                            listener       = lambda num_ready:listener.minibatch_ready(self, num_ready, dataset.train.index_sents.size)
                        )
            
            #Validation
            if dataset.val is not None:
                train_logpplx = evaluation.get_loggeomean_perplexity(self.get_sents_logprobs(max_batch_size=max_batch_size, index_sents=dataset.train.index_sents, images=dataset.train.images)[0], dataset.train.index_sents.lens)[0]
                train_logpplxs.append(train_logpplx)
                
                val_logpplx = evaluation.get_loggeomean_perplexity(self.get_sents_logprobs(max_batch_size=max_batch_size, index_sents=dataset.val.index_sents, images=dataset.val.images)[0], dataset.val.index_sents.lens)[0]
                val_logpplxs.append(val_logpplx)

                if val_logpplx >= best_val_logpplx:
                    num_times_val_logpplx_worse += 1
                else:
                    num_times_val_logpplx_worse = 0
                    best_val_logpplx = val_logpplx
                    self.save_params(param_save_dir)
                
            listener.epoch_ready(self, epoch, train_logpplx, val_logpplx)
            if early_stop_patience is not None and num_times_val_logpplx_worse >= early_stop_patience:
                break
                
        if dataset.val is not None:
            self.set_params(self.load_params(param_save_dir))
        
        listener.fit_ready(self)

        return {
                'best_val_logpplx': best_val_logpplx,
                'num_epochs': epoch,
                'stopped_early': num_times_val_logpplx_worse > 0,
                'val_logpplxs': val_logpplxs,
                'train_logpplxs': train_logpplxs,
            }
    
    
    ############################################
    def get_prefix_params(self, vocab):
        return PrefixParams(vocab, {name: val for (name, val) in self.get_params().items() if name.startswith('nn/prefix/')})
    
    
    ############################################
    @staticmethod
    def get_saved_prefix_params(vocab, save_dir):
        return PrefixParams(vocab, {name: val for (name, val) in TradNeuralModel.load_params(save_dir).items() if name.startswith('nn/prefix/')})
    
    
    ############################################
    def set_prefix_params(self, prefix_params):
        self.set_params(prefix_params.params)
    
    
    ############################################
    def _raw_run(self, nodes, max_batch_size, prefixes=None, prefixes_lens=None, images=None, targets=None, temperature=None, dropout=None, new_tokens=None, curr_states=None, shuffled=False, check_loss_nan=False, max_batches=None, listener=lambda num_ready:None):
        if prefixes is not None and type(prefixes) != np.ndarray:
            prefixes = np.array(prefixes, np.int32)
        if prefixes_lens is not None and type(prefixes_lens) != np.ndarray:
            prefixes_lens = np.array(prefixes_lens, np.int32)
        if images is not None and type(images) != np.ndarray:
            images = np.array(images, np.float32)
        if targets is not None and type(targets) != np.ndarray:
            targets = np.array(targets, np.int32)
        if new_tokens is not None and type(new_tokens) != np.ndarray:
            new_tokens = np.array(new_tokens, np.int32)
        if curr_states is not None and type(curr_states) != np.ndarray:
            curr_states = np.array(curr_states, np.float32)
        
        if images is not None and self.normalize_image == True:
            images = images/np.reshape(np.linalg.norm(images, ord=2, axis=1), [ -1, 1 ]) #reshape is to divide the images row-wise instead of column-wise
        
        num_ready = 0
        num_items = len(images) if images is not None else len(prefixes) if prefixes is not None else len(curr_states) if curr_states is not None else max_batch_size
        num_batches = int(np.ceil(num_items/max_batch_size))
        all_indexes = np.arange(num_items)
        if shuffled == True:
            np.random.shuffle(all_indexes)
        
        for batch_pos in range(num_batches if max_batches is None else min(max_batches, num_batches)):
            batch_indexes = all_indexes[batch_pos*max_batch_size:(batch_pos+1)*max_batch_size]
            batch_size = len(batch_indexes)
            batch_result = self.session.run(
                    nodes + ([self.is_loss_nan] if check_loss_nan else []),
                    feed_dict={
                            placeholder: value
                            for (placeholder, value) in [
                                    (self.batch_size,    batch_size),
                                    (self.prefixes,      prefixes[batch_indexes] if prefixes is not None else None),
                                    (self.prefixes_lens, prefixes_lens[batch_indexes] if prefixes_lens is not None else None),
                                    (self.images,        images[batch_indexes] if images is not None and self.architecture != 'langmod' else None),
                                    (self.dropout,       dropout),
                                    (self.temperature,   temperature),
                                    (self.targets,       targets[batch_indexes] if targets is not None else None),
                                    (self.new_tokens,    new_tokens[batch_indexes] if new_tokens is not None else None),
                                    (self.curr_states,   curr_states[batch_indexes] if curr_states is not None else None),
                                ]
                            if value is not None
                        }
                )
            
            if check_loss_nan:
                if batch_result[-1] == True:
                    raise NotANumberError()
                batch_result.pop()
            
            for _ in range(batch_size):
                num_ready += 1
                listener(num_ready)
                
            yield batch_result
    
    
    ############################################
    def _raw_run_whole(self, nodes, max_batch_size, prefixes=None, prefixes_lens=None, images=None, targets=None, temperature=None, dropout=None, new_tokens=None, curr_states=None, shuffled=False, check_loss_nan=False, max_batches=None, listener=lambda num_ready:None):
        all_batches = list()
        
        for batch_result in self._raw_run(nodes, max_batch_size, prefixes, prefixes_lens, images, targets, temperature, dropout, new_tokens, curr_states, shuffled, check_loss_nan, max_batches, listener):
            all_batches.append(batch_result)
        
        return [np.concatenate([ batch[i] for batch in all_batches ], axis=0) for i in range(len(nodes))]
    
    
    ############################################
    def _raw_run_iterated(self, nodes, max_batch_size, prefixes=None, prefixes_lens=None, images=None, targets=None, temperature=None, dropout=None, new_tokens=None, curr_states=None, shuffled=False, check_loss_nan=False, max_batches=None, listener=lambda num_ready:None):
        for batch_result in self._raw_run(nodes, max_batch_size, prefixes, prefixes_lens, images, targets, temperature, dropout, new_tokens, curr_states, shuffled, check_loss_nan, max_batches, listener):
            for row in batch_result:
                yield row
    
    
    ############################################
    def get_predictions(self, max_batch_size, prefixes, prefixes_lens, images, temperature, listener=lambda num_ready:None):
        return self._raw_run_whole([ self.predictions ], max_batch_size=max_batch_size, prefixes=prefixes, prefixes_lens=prefixes_lens, images=images, temperature=temperature, dropout=False, listener=listener)[0]
    
    
    ############################################
    def get_target_predictions(self, max_batch_size, prefixes, prefixes_lens, images, targets, temperature, listener=lambda num_ready:None):
        return self._raw_run_whole([ self.target_predictions ], max_batch_size=max_batch_size, prefixes=prefixes, prefixes_lens=prefixes_lens, images=images, targets=targets, temperature=temperature, dropout=False, listener=listener)[0]
    
    
    ############################################
    def get_initial_states(self, max_batch_size, images, listener=lambda num_ready:None):
        if self.architecture == 'init' or self.architecture == 'pre':
            return self._raw_run_whole([ self.init_states ], max_batch_size=max_batch_size, images=images, dropout=False, listener=listener)[0]
        else:
            return self._raw_run_whole([ self.init_states ], max_batch_size=max_batch_size, dropout=False, listener=listener)[0]
    
    
    ############################################
    def get_streamed_predictions(self, max_batch_size, new_tokens, prefixes_lens, curr_states, images, temperature, listener=lambda num_ready:None):
        return self._raw_run_whole([ self.stream_predictions, self.new_states ], max_batch_size=max_batch_size, new_tokens=new_tokens, curr_states=curr_states, images=images, temperature=temperature, dropout=False, listener=listener)
    
    
    ############################################
    def get_states(self, max_batch_size, prefixes, prefixes_lens, images, listener=lambda num_ready:None):
        result = self._raw_run_whole([ self.multimodal_vectors ], max_batch_size=max_batch_size, prefixes=prefixes, prefixes_lens=prefixes_lens, images=images, dropout=False, listener=listener)[0]
        if self.architecture == 'merge' or self.architecture == 'merge-ext':
            return result[:, :, -self.rnn_size:]
        else:
            return result
    
    
    ############################################
    def train_params(self, max_batch_size, prefixes, prefixes_lens, images, targets, max_batches=None, listener=lambda num_ready:None):
        for batch in self._raw_run([ self.train_step ], max_batch_size=max_batch_size, prefixes=prefixes, prefixes_lens=prefixes_lens, images=images, targets=targets, shuffled=True, dropout=True, check_loss_nan=True, max_batches=max_batches, listener=listener):
            pass
    
    
    ############################################
    def get_sensitivity(self, prefixes, prefixes_lens, images, targets, temperature, listener=lambda num_ready:None):
        return self._raw_run_whole([
                ]+([self.grad_nextpred_wrt_image, self.grad_maxpred_wrt_image] if self.architecture != 'langmod' else []) + [
                self.grad_nextpred_wrt_prefix, self.grad_maxpred_wrt_prefix,
                self.grad_nextpred_wrt_prevtoken, self.grad_maxpred_wrt_prevtoken,
                self.grad_nextpred_wrt_firsttoken, self.grad_maxpred_wrt_firsttoken,
                ]+([self.grad_nextpred_wrt_postimage, self.grad_maxpred_wrt_postimage] if self.architecture != 'langmod' else []) + [
                self.grad_nextpred_wrt_multimodalvec, self.grad_maxpred_wrt_multimodalvec,
            ], max_batch_size=1, prefixes=prefixes, prefixes_lens=prefixes_lens, images=images, targets=targets, temperature=temperature, dropout=False, listener=listener)
        
        
    ############################################
    def get_hidden_layers(self, max_batch_size, prefixes, prefixes_lens, images, temperature, listener=lambda num_ready:None):
        return self._raw_run_whole([
                self.post_images,
                self.last_multimodal_vectors,
                self.last_logits,
                self.last_predictions,
            ], max_batch_size=max_batch_size, prefixes=prefixes, prefixes_lens=prefixes_lens, images=images, temperature=temperature, dropout=False, listener=listener)
    