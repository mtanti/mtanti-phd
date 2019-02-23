import numpy as np
import tensorflow as tf
import h5py

from . import lib
from . import model
from . import evaluation

from .model import EmptyBeamError

########################################################################################
class NotANumberError(ArithmeticError):

    def __init__(self):
        super(NotANumberError, self).__init__()

########################################################################################
class FitListener(object):

    def __init__(self):
        pass
    
    def fit_started(self, model):
        pass
        
    def epoch_started(self, model, epoch_num):
        pass
        
    def minibatch_ready(self, model, items_ready, num_items):
        pass
    
    def epoch_ready(self, model, epoch_num, train_logpplx, val_logpplx):
        pass
    
    def fit_ready(self, model):
        pass

########################################################################################
class NeuralModel(model.Model):

    ############################################
    def __init__(self):
        super(NeuralModel, self).__init__()
        self.graph         = tf.Graph()
        self.session       = tf.Session(graph=self.graph)
        self.initializer   = None
        self.param_setters = dict()
        
    ############################################
    def compile_model(self):
        pass

    ############################################
    def set_session(self, session):
        pass
    
    ############################################
    def init_params(self):
        pass
    
    ############################################
    def get_params(self):
        with self.graph.as_default():
            params_vars = list(tf.trainable_variables())
            return { var.name: val for (var, val) in zip(params_vars, self.session.run(params_vars)) }
    
    ############################################
    def save_params(self, save_dir):
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('<tf_version>', data=np.array(tf.VERSION, np.string_))
            for (name, val) in self.get_params().items():
                f.create_dataset(name.replace('/', ' '), data=val)
    
    ############################################
    def set_params(self, new_params):
        '''
        Note that Tensorflow v1.0-1.1 used 'weights' and 'biases' as variable names in RNNs whilst later versions use 'kernel' and 'bias' instead. Update new_params if you're setting parameters across versions.
        Check Fensorflow version used to save a file using get_saved_param_version.
        '''
        with self.graph.as_default():
            param_names = list(new_params.keys())
            self.session.run(
                    [ self.param_setters[name][0] for name in param_names ],
                    { self.param_setters[name][1]: new_params[name] for name in param_names }
                )
    
    
    ############################################
    @staticmethod
    def get_saved_param_version(save_dir):
        with h5py.File(save_dir, 'r') as f:
            return str(np.array(f['<tf_version>']))
            
    
    ############################################
    @staticmethod
    def load_params(save_dir):
        with h5py.File(save_dir, 'r') as f:
            return { name.replace(' ', '/'): np.array(val) for (name, val) in f.items() if name != '<tf_version>' }
            
    
    ############################################
    def fit(self, dataset, param_save_dir, max_batch_size, minibatch_size, max_epochs, early_stop_patience=None, listener=FitListener()):
        pass
    
    ############################################
    def __exit__(self, type, value, traceback):
        if self.session is not None:
            self.session.close()