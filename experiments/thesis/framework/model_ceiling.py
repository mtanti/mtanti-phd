import numpy as np

from . import lib
from . import model
from . import data

########################################################################################
class CeilingModel(model.Model):

    ############################################
    def __init__(self, dataset):
        super(CeilingModel, self).__init__()
        
        self.oracle = dict()
        for (img, target) in zip(dataset.test.images, dataset.test.index_sents.targets):
            img_ = tuple(img)
            
            #unpad
            edge_pos = 0
            while target[edge_pos] != data.Vocab.EDGE_INDEX:
                edge_pos += 1
            target = tuple(target[:edge_pos+1])
            
            if img_ in self.oracle:
                self.oracle[img_].add(target)
            else:
                self.oracle[img_] = { target }
        self.vocab_size = dataset.vocab.size
    
    ############################################
    def get_predictions(self, max_batch_size, prefixes, prefixes_lens, images, temperature, listener=lambda num_ready:None):
        result = np.zeros([ prefixes.shape[0], prefixes.shape[1], self.vocab_size ])
        
        for (i, (image, prefix_len, prefix)) in enumerate(zip(images, prefixes_lens, prefixes)):
            #unpad
            last_token_pos = len(prefix) - 1
            while prefix[last_token_pos] == data.Vocab.EDGE_INDEX:
                last_token_pos -= 1
            true_target = tuple(prefix[:last_token_pos+1][1:]+[data.Vocab.EDGE_INDEX])
            
            true_targets = self.oracle[tuple(image)]
            if true_target in true_targets:
                for (j, target_index) in enumerate(true_target):
                    result[i, j, target_index] = 1.0
                break
            listener(i+1)
            
        return result

    ############################################
    def get_target_predictions(self, max_batch_size, prefixes, prefixes_lens, images, targets, temperature, listener=lambda num_ready:None):
        preds = self.get_predictions(max_batch_size, prefixes, prefixes_lens, images, temperature, listener)
        return preds.reshape(
                [preds.shape[0]*preds.shape[1], preds.shape[2]]
            )[
                np.arange(preds.shape[0]*preds.shape[1]),
                targets.reshape([preds.shape[0]*preds.shape[1]])
            ].reshape([preds.shape[0], preds.shape[1]])
        
    ############################################
    def get_initial_states(self, max_batch_size, images, listener=lambda num_ready:None):
        return [ None ]*len(images)
    
    ############################################
    def get_streamed_predictions(self, max_batch_size, new_tokens, prefixes_lens, curr_states, images, temperature, listener=lambda num_ready:None):
        result = np.zeros([ len(images), self.vocab_size ])
        
        for (i, (image, prefix_len)) in enumerate(zip(images, prefixes_lens)):
            true_target = next(iter(self.oracle[tuple(image)]))
            if prefix_len - 1 < len(true_target):
                result[i, true_target[prefix_len - 1]] = 1.0
            listener(i+1)
            
        return [ result, [ None ]*len(images) ]