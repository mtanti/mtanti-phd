import numpy as np
import collections
import heapq

from . import lib
from . import data

########################################################################################
class EmptyBeamError(ArithmeticError):

    def __init__(self):
        super(EmptyBeamError, self).__init__()

#################################################################
class _Beam(object):
#For use by beam search.

    #################################################################
    def __init__(self, beam_width):
        self.heap = list()
        self.next_id = 0
        self.beam_width = beam_width
    
    #################################################################
    def add(self, prefix_logprob, complete, prefix, prefix_len, curr_state):
        heapq.heappush(self.heap, (prefix_logprob, self.next_id, (complete, prefix, prefix_len, curr_state))) #use a tuple comparison barrier to guarantee that the value items are not used for comparison is the logprob is equal
        self.next_id += 1
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)
            
    #################################################################
    def can_add(self, new_prefix_logprob):
        if len(self.heap) < self.beam_width:
            return True
        else:
            return new_prefix_logprob > self.heap[0][0]
    
    #################################################################
    def __len__(self):
        return len(self.heap)
    
    #################################################################
    def __getitem__(self, i):
        (prefix_logprob, _, items) = self.heap[i]
        return (prefix_logprob, items)
    
    #################################################################
    def __iter__(self):
        return ((prefix_logprob, items) for (prefix_logprob, _, items) in self.heap)

########################################################################################
class Model(object):

    ############################################
    def __init__(self):
        pass
        
    ############################################
    def compile_model(self):
        pass

    ############################################
    def init_params(self):
        pass
    
    ############################################
    def save_params(self, param_save_dir):
        pass
    
    ############################################
    def load_params(self, param_save_dir):
        pass
    
    ############################################
    def get_num_params(self):
        return 0
    
    ############################################
    def get_predictions(self, max_batch_size, prefixes, prefixes_lens, images, temperature, listener=lambda num_ready:None):
        pass
    
    ############################################
    def get_target_predictions(self, max_batch_size, prefixes, prefixes_lens, images, targets, temperature, listener=lambda num_ready:None):
        pass
    
    ############################################
    def get_initial_states(self, max_batch_size, images, listener=lambda num_ready:None):
        pass
    
    ############################################
    def get_streamed_predictions(self, max_batch_size, new_tokens, prefixes_lens, curr_states, images, temperature, listener=lambda num_ready:None):
        pass
    
    ############################################
    def get_tokens_logprobs(self, max_batch_size, index_sents, images=None, listener=lambda num_ready:None):
        if index_sents.targets is None:
            raise('index_sents must have target sentences.')
            
        target_probs = self.get_target_predictions(
                max_batch_size = max_batch_size,
                prefixes       = index_sents.prefixes,
                prefixes_lens  = index_sents.lens,
                targets        = index_sents.targets,
                images         = images if images is not None else None,
                temperature    = 1.0,
                listener       = listener
            )
        
        return [
                [ np.log2(p) if p != 0.0 else -np.inf for p in token_probs[:sent_len] ]
                for (token_probs, sent_len) in zip(target_probs, index_sents.lens)
            ]
        
    ############################################
    def get_sents_logprobs(self, max_batch_size, index_sents, images=None, listener=lambda num_ready:None):
        tokens_logprobs = self.get_tokens_logprobs(max_batch_size, index_sents, images, listener)
        return ([
                np.sum(token_logprobs) if -np.inf not in token_logprobs else -np.inf
                for token_logprobs in tokens_logprobs
            ], tokens_logprobs)
    
    ############################################
    def generate_sents_sample(self, max_batch_size, images, lower_bound_len=3, upper_bound_len=50, temperature=1.0, listener=lambda num_ready:None):
        amount = len(images)
        is_sent_complete        = [ False ]*amount
        complete_sents_logprobs = [ None ]*amount
        complete_sents_prefixes = [ None ]*amount
        complete_sents_lens     = [ None ]*amount
        
        prev_beams = []
        init_states = self.get_initial_states(amount, images)
        for init_state in init_states:
            beam = _Beam(1)
            beam.add(0.0, False, [ data.Vocab.EDGE_INDEX ]*(upper_bound_len + 1), 1, init_state)
            prev_beams.append(beam)
        
        num_ready = 0
        
        while True:
            curr_beams = [ _Beam(1) for _ in range(amount) ]
            batch_orig_indexes = []
            batch_prefixes_logprobs = []
            batch_prefixes = []
            batch_prefixes_lens = []
            batch_new_tokens = []
            batch_curr_states = []
            batch_images = []
            beams_batched = 0
            for i in range(amount):
                if is_sent_complete[i] == False:
                    (prefix_logprob, (complete, prefix, prefix_len, curr_state)) = prev_beams[i][0]
                    if complete == True:
                        curr_beams[i].add(prefix_logprob, True, prefix, prefix_len, curr_state)
                    else:
                        batch_orig_indexes.append(i)
                        batch_prefixes_logprobs.append(prefix_logprob)
                        batch_prefixes.append(prefix)
                        batch_prefixes_lens.append(prefix_len)
                        batch_new_tokens.append(prefix[prefix_len - 1])
                        batch_curr_states.append(curr_state)
                        batch_images.append(images[i])
                    beams_batched += 1
                    
            if beams_batched == 0:
                return (data.IndexSents(np.array(complete_sents_prefixes, np.int32), complete_sents_lens), complete_sents_logprobs)
                
            [ batch_distributions, batch_new_states ] = self.get_streamed_predictions(
                    max_batch_size = max_batch_size,
                    new_tokens     = batch_new_tokens,
                    curr_states    = batch_curr_states,
                    prefixes_lens  = batch_prefixes_lens,
                    images         = batch_images,
                    temperature    = temperature,
                )
            
            grouped_beam_batches = dict()
            for (orig_index, prefix_logprob, prefix, prefix_len, new_state, distribution) in zip(batch_orig_indexes, batch_prefixes_logprobs, batch_prefixes, batch_prefixes_lens, batch_new_states, batch_distributions):
                grouped_beam_batches[orig_index] = (prefix_logprob, prefix, prefix_len, new_state, distribution)
                
            for (orig_index, beam) in grouped_beam_batches.items():
                (prefix_logprob, prefix, prefix_len, new_state, distribution) = beam
                prev_token_index = prefix[prefix_len - 1]
                
                unknown_prob = distribution[data.Vocab.UNKNOWN_INDEX]
                prev_token_prob = distribution[prev_token_index]
                edge_prob = distribution[data.Vocab.EDGE_INDEX]
                
                full_total_prob = 1.0 - unknown_prob
                if prefix_len > 1:
                    full_total_prob -= prev_token_prob
                if prefix_len - 1 < lower_bound_len:
                    full_total_prob -= edge_prob
                
                rand_total_prob = np.random.random()*full_total_prob
                running_total_prob = 0.0
                for (next_index, next_prob) in enumerate(distribution.tolist()):
                    if next_prob == 0.0:
                        pass
                    elif next_index == data.Vocab.UNKNOWN_INDEX: #skip unknown
                        pass
                    elif next_index == prefix[prefix_len - 1]: #skip repeating words
                        pass
                    elif next_index == data.Vocab.EDGE_INDEX and prefix_len - 1 < lower_bound_len: #if next item is the end token then mark prefix as complete and leave out the end token
                        pass
                    else:
                        running_total_prob += next_prob
                        if running_total_prob >= rand_total_prob:
                            if next_index == data.Vocab.EDGE_INDEX: #if next item is the end token then mark prefix as complete and leave out the end token
                                curr_beams[orig_index].add(prefix_logprob + np.log2(next_prob), True, prefix, prefix_len, new_state)
                            else: #if next item is a non-end token then mark prefix as incomplete (if its length does not exceed the clip length, ignoring start token)
                                new_prefix = list(prefix)
                                new_prefix[prefix_len] = next_index
                                curr_beams[orig_index].add(prefix_logprob + np.log2(next_prob), prefix_len == upper_bound_len, new_prefix, prefix_len + 1, new_state) #when checking if sentence is complete, check if the length of the new prefix minus the edge token equal the upper bound length
                            break
                
                if len(curr_beams[orig_index]) == 0:
                    raise EmptyBeamError()
                
                (prefix_logprob, (complete, prefix, prefix_len, curr_state)) = curr_beams[orig_index][0]
                if complete == True:
                    is_sent_complete[orig_index] = True
                    curr_beams[orig_index] = None
                    complete_sents_logprobs[orig_index] = prefix_logprob
                    complete_sents_prefixes[orig_index] = prefix
                    complete_sents_lens[orig_index] = prefix_len
                    num_ready += 1
                    listener(num_ready)
                    
                prev_beams[orig_index] = curr_beams[orig_index]
    
    ############################################
    def generate_sents_beamsearch(self, max_batch_size, images, beam_width=3, lower_bound_len=3, upper_bound_len=50, temperature=1.0, listener=lambda num_ready:None):
        amount = len(images)
        is_sent_complete        = [ False ]*amount
        complete_sents_logprobs = [ None ]*amount
        complete_sents_prefixes = [ None ]*amount
        complete_sents_lens     = [ None ]*amount
        
        prev_beams = []
        init_states = self.get_initial_states(amount, images)
        for init_state in init_states:
            beam = _Beam(beam_width)
            beam.add(0.0, False, [ data.Vocab.EDGE_INDEX ]*(upper_bound_len + 1), 1, init_state)
            prev_beams.append(beam)
        
        num_ready = 0
        
        while True:
            curr_beams = [ _Beam(beam_width) for _ in range(amount) ]
            batch_orig_indexes = []
            batch_prefixes_logprobs = []
            batch_prefixes = []
            batch_prefixes_lens = []
            batch_new_tokens = []
            batch_curr_states = []
            batch_images = []
            beams_batched = 0
            for i in range(amount):
                if is_sent_complete[i] == False:
                    for (prefix_logprob, (complete, prefix, prefix_len, curr_state)) in prev_beams[i]:
                        if complete == True:
                            curr_beams[i].add(prefix_logprob, True, prefix, prefix_len, curr_state)
                        else:
                            batch_orig_indexes.append(i)
                            batch_prefixes_logprobs.append(prefix_logprob)
                            batch_prefixes.append(prefix)
                            batch_prefixes_lens.append(prefix_len)
                            batch_new_tokens.append(prefix[prefix_len - 1])
                            batch_curr_states.append(curr_state)
                            batch_images.append(images[i])
                    beams_batched += 1
                    
            if beams_batched == 0:
                return (data.IndexSents(np.array(complete_sents_prefixes, np.int32), complete_sents_lens), complete_sents_logprobs)
                
            [ batch_distributions, batch_new_states ] = self.get_streamed_predictions(
                    max_batch_size = max_batch_size,
                    new_tokens     = batch_new_tokens,
                    curr_states    = batch_curr_states,
                    prefixes_lens  = batch_prefixes_lens,
                    images         = batch_images,
                    temperature    = temperature,
                )
            
            grouped_beam_batches = collections.defaultdict(list)
            for (orig_index, prefix_logprob, prefix, prefix_len, new_state, distribution) in zip(batch_orig_indexes, batch_prefixes_logprobs, batch_prefixes, batch_prefixes_lens, batch_new_states, batch_distributions):
                grouped_beam_batches[orig_index].append((prefix_logprob, prefix, prefix_len, new_state, distribution))
                
            for (orig_index, beam_group) in grouped_beam_batches.items():
                for (prefix_logprob, prefix, prefix_len, new_state, distribution) in beam_group:
                    for (next_index, next_prob) in enumerate(distribution.tolist()):
                        if next_prob == 0.0:
                            pass
                        elif next_index == data.Vocab.UNKNOWN_INDEX: #skip unknown
                            pass
                        elif next_index == prefix[prefix_len - 1]: #skip repeating words
                            pass
                        elif next_index == data.Vocab.EDGE_INDEX and prefix_len - 1 < lower_bound_len: #only consider terminating the prefix if it has sufficient length after ignoring the edge token
                            pass
                        elif next_index == data.Vocab.EDGE_INDEX: #if next item is the end token then mark prefix as complete and leave out the end token
                            new_prefix_logprob = prefix_logprob + np.log2(next_prob)
                            if curr_beams[orig_index].can_add(new_prefix_logprob):
                                curr_beams[orig_index].add(new_prefix_logprob, True, prefix, prefix_len, new_state)
                        else: #if next item is a non-end token then mark prefix as incomplete (if its length does not exceed the clip length, ignoring start token)
                            new_prefix_logprob = prefix_logprob + np.log2(next_prob)
                            if curr_beams[orig_index].can_add(new_prefix_logprob):
                                new_prefix = list(prefix)
                                new_prefix[prefix_len] = next_index
                                curr_beams[orig_index].add(new_prefix_logprob, prefix_len == upper_bound_len, new_prefix, prefix_len + 1, new_state) #when checking if sentence is complete, check if the length of the new prefix minus the edge token equal the upper bound length
                
                if len(curr_beams[orig_index]) == 0:
                    raise EmptyBeamError()
                (best_logprob, (best_complete, best_prefix, best_prefix_len, best_curr_state)) = max(curr_beams[orig_index], key=lambda x:x[0]) #when getting the max heap item, ignore anything other than the logprob (otherwise it will use other items for coincidentally equal logprobed items)
                
                if best_complete == True:
                    is_sent_complete[orig_index] = True
                    curr_beams[orig_index] = None
                    complete_sents_logprobs[orig_index] = best_logprob
                    complete_sents_prefixes[orig_index] = best_prefix
                    complete_sents_lens[orig_index] = best_prefix_len
                    num_ready += 1
                    listener(num_ready)

                prev_beams[orig_index] = curr_beams[orig_index]
    
    ############################################
    def __enter__(self):
        return self
    
    ############################################
    def __exit__(self, type, value, traceback):
        pass
