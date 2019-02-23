import collections
import numpy as np
import json

from . import config

################################################################################
class Vocab(object):
    
    (EDGE_INDEX, EDGE_TOKEN)       = (0, '<EDG>')
    (UNKNOWN_INDEX, UNKNOWN_TOKEN) = (1, '<UNK>')

    ############################################
    def __init__(self, vocab_list):
        assert vocab_list[Vocab.EDGE_INDEX] == Vocab.EDGE_TOKEN
        assert vocab_list[Vocab.UNKNOWN_INDEX] == Vocab.UNKNOWN_TOKEN
        
        self.vocab_list     = vocab_list
        self.vocab_set      = set(vocab_list)
        self.size           = len(vocab_list)
        self.token_to_index = { token: i for (i, token) in enumerate(vocab_list) }
        self.index_to_token = { i: token for (i, token) in enumerate(vocab_list) }
    
    ############################################
    def tokens_to_indexes(self, tokens):
        return [ self.token_to_index.get(token, Vocab.UNKNOWN_INDEX) for token in tokens ]
    
    ############################################
    def indexes_to_tokens(self, indexes):
        return [ self.index_to_token[index] for index in indexes ]
    
    ############################################
    def intersection(self, vocab):
        return Vocab([Vocab.EDGE_TOKEN, Vocab.UNKNOWN_TOKEN] + sorted((self.vocab_set & vocab.vocab_set) - {Vocab.EDGE_TOKEN, Vocab.UNKNOWN_TOKEN}))
    
    ############################################
    @staticmethod
    def load_vocab(save_dir):
        with open(save_dir, 'r', encoding='utf-8') as f:
            vocab_list = json.load(f)
        
        return Vocab(vocab_list)
    
    ############################################
    def save_vocab(self, save_dir):
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(self.vocab_list, f)


################################################################################
class TextSents(object):

    ############################################
    def __init__(self, sents):
        self.sents   = sents
        self.size    = len(sents)
        self.max_len = max(len(sent) for sent in sents)
    
    ############################################
    def get_vocab_freqs(self):
        all_tokens = (token for sent in self.sents for token in sent)
        return collections.Counter(all_tokens)
    
    ############################################
    def get_vocab(self, min_token_freq=None, top_tokens=None):
        token_freqs = self.get_vocab_freqs()
        vocab = sorted(token_freqs.keys(), key=lambda token:(-token_freqs[token], token))
        if min_token_freq is not None:
            while token_freqs[vocab[-1]] < min_token_freq:
                vocab.pop()
        if top_tokens is not None:
            vocab = vocab[:top_tokens]
        vocab = [ Vocab.EDGE_TOKEN, Vocab.UNKNOWN_TOKEN ] + sorted(vocab)
        
        return Vocab(vocab)
    
    ############################################
    def compile_sents(self, vocab, add_end_token=True):
        sents_indexes = list()
        lens          = list()
        for sent in self.sents:
            sents_indexes.append(vocab.tokens_to_indexes(sent))
            lens.append(len(sent)+1) #add 1 due to edge token

        prefixes_indexes = np.zeros([self.size, self.max_len+1], np.int32) + Vocab.EDGE_INDEX
        targets_indexes  = np.zeros([self.size, self.max_len+1], np.int32) + Vocab.EDGE_INDEX
        for (i, sent_indexes) in enumerate(sents_indexes):
            prefixes_indexes[i,:lens[i]] = [Vocab.EDGE_INDEX]+sent_indexes
            targets_indexes [i,:lens[i]] = sent_indexes+[Vocab.EDGE_INDEX]

        return IndexSents(prefixes_indexes, lens, targets_indexes)


################################################################################
class IndexSents(object):

    ############################################
    def __init__(self, prefixes, lens, targets=None):
        if type(prefixes) != np.ndarray:
            prefixes = np.array(prefixes, np.int32)
            
        self.prefixes = prefixes
        self.lens     = lens
        self.targets  = targets
        self.size     = prefixes.shape[0]
        self.max_len  = prefixes.shape[1]
    
    ############################################
    def add_targets(self):
        self.targets = np.zeros_like(self.prefixes, np.int32) + Vocab.EDGE_INDEX
        self.targets[:, :-1] = self.prefixes[:, 1:]
    
    ############################################
    def decompile_sents(self, vocab):
        return TextSents([ vocab.indexes_to_tokens(prefix[1:prefix_len]) for (prefix, prefix_len) in zip(self.prefixes, self.lens) ])


################################################################################
class DataSource(object):

    ############################################
    def __init__(self, raw_sents, images=None, filenames=None, group_indexes=None, individual_indexes=None, grouped_sents=False):
        if grouped_sents == True:
            self.raw_sents = [ sent for group in raw_sents for sent in group ]
            
            if images is not None:
                self.images = [ image for (image, group) in zip(images, raw_sents) for sent in group ]
            else:
                self.images = None
            
            if filenames is not None:
                self.filenames = [ filename for (filename, group) in zip(filenames, raw_sents) for sent in group ]
            else:
                self.filenames = None
            
            if group_indexes is not None:
                self.group_indexes = group_indexes
            else:
                self.group_indexes = [ index for (index, group) in enumerate(raw_sents) for sent in group ]
            
        else:
            self.raw_sents = raw_sents
            
            if images is not None:
                self.images = images
            else:
                self.images = None
                
            if filenames is not None:
                self.filenames = filenames
            else:
                self.filenames = None
            
            if group_indexes is not None:
                self.group_indexes = group_indexes
            else:
                self.group_indexes = list(range(len(raw_sents)))
        
        if individual_indexes is not None:
            self.individual_indexes = individual_indexes
        else:
            self.individual_indexes = list(range(len(self.raw_sents)))
        
        self.num_groups  = len(raw_sents)
        self.size        = len(self.raw_sents)
        self.text_sents  = None
        self.index_sents = None
    
    ############################################
    def tokenize_sents(self):
        self.text_sents = TextSents([ sent.split(' ') for sent in self.raw_sents ])
        return self
    
    ############################################
    def compile_sents(self, vocab):
        self.index_sents = self.text_sents.compile_sents(vocab)
        return self
    
    ############################################
    def shuffle(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)
        rand = np.random.RandomState()
        
        rand.seed(seed)
        rand.shuffle(self.raw_sents)
        
        if self.text_sents is not None:
            rand.seed(seed)
            rand.shuffle(self.text_sents.sents)
        
        if self.index_sents is not None:
            rand.seed(seed)
            rand.shuffle(self.index_sents.prefixes)
            rand.seed(seed)
            rand.shuffle(self.index_sents.lens)
            if self.index_sents.targets is not None:
                rand.seed(seed)
                rand.shuffle(self.index_sents.targets)
        
        if self.images is not None:
            rand.seed(seed)
            rand.shuffle(self.images)
            
        if self.filenames is not None:
            rand.seed(seed)
            rand.shuffle(self.filenames)
        
        rand.seed(seed)
        rand.shuffle(self.group_indexes)
        
        rand.seed(seed)
        rand.shuffle(self.individual_indexes)
        
        return self
    
    ############################################
    def order_by_indexes(self):
        self.raw_sents.sort(key=lambda i:self.individual_indexes[i])
        
        if self.text_sents is not None:
            self.text_sents.sort(key=lambda i:self.individual_indexes[i])
        
        if self.index_sents is not None:
            self.index_sents.sort(key=lambda i:self.individual_indexes[i])
        
        if self.images is not None:
            self.images.sort(key=lambda i:self.individual_indexes[i])
            
        if self.filenames is not None:
            self.filenames.sort(key=lambda i:self.individual_indexes[i])
        
        self.group_indexes.sort(key=lambda i:self.individual_indexes[i])
        
        self.individual_indexes.sort()
        
        return self
    
    ############################################
    def take(self, amount=None, one_per_group=False, whole_groups=False):
        if one_per_group == True and whole_groups == True:
            raise ValueError('Can\'t take one item per group as well as whole groups.')
            
        if amount is None:
            amount = self.size
            
        raw_sents          = list()
        images             = list() if self.images is not None else None
        filenames          = list() if self.filenames is not None else None
        group_indexes      = list()
        individual_indexes = list()
        
        groups_added = set()
        for (i, group_index) in enumerate(self.group_indexes):
            if one_per_group == True:
                if group_index in groups_added:
                    continue
                if len(raw_sents) >= amount:
                    break
            elif whole_groups == True:
                if group_index not in groups_added and len(groups_added) >= amount:
                    continue
            else:
                if len(raw_sents) >= amount:
                    break
            
            raw_sents.append(self.raw_sents[i])
            if self.images is not None:
                images.append(self.images[i])
            if self.filenames is not None:
                filenames.append(self.filenames[i])
            group_indexes.append(self.group_indexes[i])
            individual_indexes.append(self.individual_indexes[i])
            groups_added.add(group_index)
        
        return DataSource(
                raw_sents          = raw_sents,
                images             = images,
                filenames          = filenames,
                group_indexes      = group_indexes,
                individual_indexes = individual_indexes,
            )
    
    ############################################
    def without_images(self):
        return DataSource(
                raw_sents          = self.raw_sents,
                images             = None,
                filenames          = None,
                group_indexes      = self.group_indexes,
                individual_indexes = self.individual_indexes,
            )
    
    ############################################
    def get_filenames(self):
        if self.filenames is None:
            raise ValueError('Can\'t get filenames from a dataset with no filenames.')
            
        filenames = list()
        groups_added = set()
        for (group_index, filename) in zip(self.group_indexes, self.filenames):
            if group_index not in groups_added:
                filenames.append(filename)
                groups_added.add(group_index)
                
        return filenames
    
    ############################################
    def get_images(self):
        if self.images is None:
            raise ValueError('Can\'t get images from a dataset with no images.')
            
        images = list()
        groups_added = set()
        for (group_index, image) in zip(self.group_indexes, self.images):
            if group_index not in groups_added:
                images.append(image)
                groups_added.add(group_index)
                
        return images
        
    ############################################
    def get_text_sent_groups(self):
        groups_added = set()
        group_order = list()
        groups = dict()
        for (group_index, text_sent) in zip(self.group_indexes, self.text_sents.sents):
            if group_index not in groups_added:
                groups[group_index] = [text_sent]
                group_order.append(group_index)
                groups_added.add(group_index)
            else:
                groups[group_index].append(text_sent)
        
        return [groups[group_index] for group_index in group_order]


########################################################################################
class Dataset(object):

    ############################################
    def __init__(self, vocab, train_datasource=None, val_datasource=None, test_datasource=None):
        self.vocab = vocab
        self.train = train_datasource
        self.val   = val_datasource
        self.test  = test_datasource
    
    ############################################
    def compile_sents(self):
        if self.train is not None:
            self.train.tokenize_sents()
            self.train.compile_sents(self.vocab)
        if self.val is not None:
            self.val.tokenize_sents()
            self.val.compile_sents(self.vocab)
        if self.test is not None:
            self.test.tokenize_sents()
            self.test.compile_sents(self.vocab)

########################################################################################
def load_datasources(dataset_name):
    datasources = dict()
    
    if dataset_name == 'lm1b':
        with open(config.dataset_dir+'/lm1b.json', 'r', encoding='utf-8') as f:
            sents = json.load(f)
        
        for split in sents.keys():
            datasources[split] = DataSource(
                raw_sents=sents[split]['sents'],
                grouped_sents=False
            )
    else:
        with open(config.dataset_dir+'/'+dataset_name+'.json', 'r', encoding='utf-8') as f:
            sents = json.load(f)
        
        for split in sents.keys():
            datasources[split] = DataSource(
                raw_sents=sents[split]['sents'],
                images=np.load(config.dataset_dir+'/'+dataset_name+'_'+split+'.npy'),
                filenames=sents[split]['fnames'],
                grouped_sents=True
            )
            
    return datasources