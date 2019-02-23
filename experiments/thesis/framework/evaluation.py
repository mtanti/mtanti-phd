import sys
import numpy as np
import nltk
import collections
import json

from . import config

sys.path.append(config.mscoco_eval_dir)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.wmd.wmd import WMD

_meteor_scorer = Meteor()
_cider_scorer = Cider()
_spice_scorer = Spice()
_wmd_scorer   = WMD()

########################################################################################
def get_logperplexities(logprobs, sent_lens):
    # Let P = probability of a sentence with L words
    # Let pi = probability of word i in sentence
    # P = p1*...*pL
    # log P = (log p1) + ... + (log pL)
    # pplx = 2^(-1/L log P) = 2^(-(log p1 + ... + log pL)/L)
    # log pplx = -(log p1 + ... + log pL)/L = -(log P)/L
    return [ -logprob/sent_len for (logprob, sent_len) in zip(logprobs, sent_lens) ]

########################################################################################
def get_loggeomean_perplexity(logprobs, sent_lens):
    # Let pplxi = perplexity of sentence i out of N sentences
    # geomean = (pplx1*...*pplxN)**(1/N)
    # log geomean = (1/N) log (pplx1*...*pplxN) = (log pplx1 + ... + log pplxN)/N
    logpplxs = get_logperplexities(logprobs, sent_lens)
    return (
            np.sum(logpplx for logpplx in logpplxs if not np.isinf(logpplx))/len(logpplxs),
            sum(np.isinf(logpplx) for logpplx in logpplxs)
        )

########################################################################################
def get_probability_stats(logprobs, sent_lens, num_unknowns_per_sent=None, num_out_of_vocab_tokens=None):
    if num_unknowns_per_sent is not None:
        #Since the unknown token stands in place of every out of vocabulary word, the more out of vocabulary words in the sentences the greater the unknown token's probability (in the limiting case if every word is an out of vocabulary word then all words have a probability of 1).
        #To compensate for this we shall assume that each different out of vocabulary word has an equal share of the unknown token's probability by dividing the unknown token's probability by the number of out of vocabulary words.
        # P = p1*...*pUNK*...pi*...*pUNK*...*pL
        # P' = p1*...*(pUNK/#oov)*...pi*...*(pUNK/#oov)*...*pL
        # P' = (p1*...*pUNK*...pi*...*pUNK*...*pL)/(#oov^#unk)
        # P' = P/(#oov^#unk)
        # log P' = log P - #unk*(log #oov)
        logprobs = [ logprob - num_unknowns*np.log2(num_out_of_vocab_tokens) for (logprob, num_unknowns) in zip(logprobs, num_unknowns_per_sent) ]
    probs = [ 2**logprob for logprob in logprobs ]
    logpplxs = get_logperplexities(logprobs, sent_lens)
    pplxs = [ 2**logpplx for logpplx in logpplxs ]
    return {
        'mean_prob': np.mean(probs),
        'median_prob': np.median(probs),
        'geomean_prob': 2**np.mean(logprobs),
        'mean_pplx': np.mean(pplxs),
        'median_pplx': np.median(pplxs),
        'geomean_pplx': 2**np.mean(logpplxs)
    }

########################################################################################
def get_meteor_score(test_tokenized_grouped_sents, generated):
    return _meteor_scorer.compute_score(
        {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
        {i: [ ' '.join(g) ] for (i, g) in enumerate(generated)}
    )

########################################################################################
def get_cider_score(test_tokenized_grouped_sents, generated):
    return _cider_scorer.compute_score(
        {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
        {i: [ ' '.join(g) ] for (i, g) in enumerate(generated)}
    )

########################################################################################
def get_spice_score(test_tokenized_grouped_sents, generated):
    return _spice_scorer.compute_score(
        {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
        {i: [ ' '.join(g) ] for (i, g) in enumerate(generated)}
    )

########################################################################################
def get_wmd_score(test_tokenized_grouped_sents, generated):
    return _wmd_scorer.compute_score(
        {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
        {i: [ ' '.join(g) ] for (i, g) in enumerate(generated)}
    )

########################################################################################
def mscoco_eval(test_tokenized_sent_groups, generated_sents_tokenized):
    with open(config.mscoco_eval_dir+'/annotations/references.json', 'w', encoding='utf-8') as f:
        json.dump({
                'info': {'description': None, 'url': None, 'version': None, 'year': None, 'contributor': None, 'date_created': None},
                'images': [
                        {'license': None, 'url': None, 'file_name': None, 'id': image_id, 'width': None, 'date_captured': None, 'height': None}
                        for image_id in range(len(test_tokenized_sent_groups))
                    ],
                'licenses': [],
                'type': 'captions',
                'annotations': [
                        {
                            'image_id': image_id,
                            'id':       caption_id,
                            'caption':  ' '.join(sent)
                        }
                        for (caption_id, (image_id, sent)) in enumerate(
                            (image_id, sent)
                            for (image_id, sent_group) in enumerate(test_tokenized_sent_groups)
                            for sent in sent_group
                        )
                    ]
            }, f)
            
    with open(config.mscoco_eval_dir+'/results/generated.json', 'w', encoding='utf-8') as f:
        json.dump([
            {
                'image_id': image_id,
                'caption':  ' '.join(sent)
            }
            for (image_id, sent) in enumerate(generated_sents_tokenized)
        ], f)
        
    coco = COCO(config.mscoco_eval_dir+'/annotations/references.json')
    cocoRes = coco.loadRes(config.mscoco_eval_dir+'/results/generated.json')
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    return {
            'Bleu_1': cocoEval.eval['Bleu_1'],
            'Bleu_2': cocoEval.eval['Bleu_2'],
            'Bleu_3': cocoEval.eval['Bleu_3'],
            'Bleu_4': cocoEval.eval['Bleu_4'],
            'METEOR': cocoEval.eval['METEOR'],
            'ROUGE_L': cocoEval.eval['ROUGE_L'],
            'CIDEr': cocoEval.eval['CIDEr'],
            'SPICE': cocoEval.eval['SPICE'],
            'WMD': cocoEval.eval['WMD'],
            'Bleu_1_all': [ item['Bleu_1'] for item in cocoEval.evalImgs ],
            'Bleu_2_all': [ item['Bleu_2'] for item in cocoEval.evalImgs ],
            'Bleu_3_all': [ item['Bleu_3'] for item in cocoEval.evalImgs ],
            'Bleu_4_all': [ item['Bleu_4'] for item in cocoEval.evalImgs ],
            'METEOR_all': [ item['METEOR'] for item in cocoEval.evalImgs ],
            'ROUGE_L_all': [ item['ROUGE_L'] for item in cocoEval.evalImgs ],
            'CIDEr_all': [ item['CIDEr'] for item in cocoEval.evalImgs ],
            'SPICE_all': [ item['SPICE']['All']['f'] for item in cocoEval.evalImgs ],
            'WMD_all': [ item['WMD'] for item in cocoEval.evalImgs ],
        }

########################################################################################
def diversity_eval(train_tokenized_sents, test_tokenized_grouped_sents, vocab, train_token_freqs, tokenized_generated_sents):
    known_train_sents_full   = { ' '.join(sent) for sent in train_tokenized_sents },
    known_train_sents_3grams = { ' '.join(sent[i:i+3]) for sent in train_tokenized_sents for i in range(len(sent)-3+1) }
    known_train_sents_4grams = { ' '.join(sent[i:i+4]) for sent in train_tokenized_sents for i in range(len(sent)-4+1) }
    known_train_sents_5grams = { ' '.join(sent[i:i+5]) for sent in train_tokenized_sents for i in range(len(sent)-5+1) }
    test_grouped_sents = [ [ ' '.join(sent) for sent in group ] for group in test_tokenized_grouped_sents ]
    
    unique_sents           = set()
    num_reused_sents       = 0
    reused_sents           = list()
    reused_sent_test_sents = list()
    num_reused_3grams      = 0
    num_reused_4grams      = 0
    num_reused_5grams      = 0
    sent_lens              = list()
    tagged_tokens          = collections.defaultdict(set)
    token_unigrams         = set()
    token_bigrams          = set()
    token_trigrams         = set()
    for (generated_sent_tokens, test_sents) in zip(tokenized_generated_sents, test_grouped_sents):
        generated_sent = ' '.join(generated_sent_tokens)
        token_tags = nltk.pos_tag(generated_sent_tokens, tagset='universal')
        tags = [tag for (token, tag) in token_tags]
        sent_len = len(generated_sent_tokens)
        
        sent_lens.append(sent_len)
        
        unique_sents.add(generated_sent)
        
        if generated_sent in known_train_sents_full:
            num_reused_sents += 1
            reused_sents.append(generated_sent)
            reused_sent_test_sents.append(test_sents)
            
        for (token, tag) in token_tags:
            tagged_tokens[tag].add(token)
            
        for i in range(len(generated_sent_tokens)):
            if i < sent_len-0:
                token_unigrams.add(tuple(generated_sent_tokens[i:i+1]))
            if i < sent_len-1:
                token_bigrams.add(tuple(generated_sent_tokens[i:i+2]))
            if i < sent_len-2:
                token_trigrams.add(tuple(generated_sent_tokens[i:i+3]))
        
        for i in range(len(generated_sent_tokens)):
            if i < sent_len-2:
                if ' '.join(generated_sent_tokens[i:i+3]) in known_train_sents_3grams:
                    num_reused_3grams += 1
            if i < sent_len-3:
                if ' '.join(generated_sent_tokens[i:i+4]) in known_train_sents_4grams:
                    num_reused_4grams += 1
            if i < sent_len-4:
                if ' '.join(generated_sent_tokens[i:i+5]) in known_train_sents_5grams:
                    num_reused_5grams += 1
            
    num_vocab_used = sum((token in vocab.vocab_set) for (token,) in token_unigrams) #Filtered for the ceiling model sentences which would have all words in the test set rather than words in the vocabulary
    
    min_freq_vocab_used = min(train_token_freqs[token] for (token,) in token_unigrams if token in vocab.vocab_set)
    
    reused_sents_wmd = get_wmd_score(reused_sent_test_sents, reused_sents)[0] if num_reused_sents > 0 else None
    
    return {
            'vocab_used': num_vocab_used,
            'vocab_used_frac': num_vocab_used/vocab.size,
            'min_freq_vocab_used': min_freq_vocab_used,
            'min_sent_len': min(sent_lens),
            'mean_sent_len': np.mean(sent_lens),
            'max_sent_len': max(sent_lens),
            'num_reused_sents': len(reused_sents),
            'num_reused_sents_frac': len(reused_sents)/len(tokenized_generated_sents),
            'reused_sents_WMD': reused_sents_wmd,
            'num_reused_3grams': num_reused_3grams,
            'num_reused_4grams': num_reused_4grams,
            'num_reused_5grams': num_reused_5grams,
            'num_unique_sents': len(unique_sents),
            'num_unique_sents_frac': len(unique_sents)/len(tokenized_generated_sents),
            'num_types_nouns': len(tagged_tokens['NOUN']),
            'num_types_adjectives': len(tagged_tokens['ADJ']),
            'num_types_verbs': len(tagged_tokens['VERB']),
            'num_types_adverbs': len(tagged_tokens['ADV']),
            'num_types_unigrams': len(token_unigrams),
            'num_types_bigrams': len(token_bigrams),
            'num_types_trigrams': len(token_trigrams),
        }

########################################################################################
def retrieval_eval(image_caption_logprobs_matrix):
    r1  = 0
    r5  = 0
    r10 = 0
    ranks = list()
    for (correct_index, logprobs) in enumerate(image_caption_logprobs_matrix):
        retrieved_indexes = np.argsort(logprobs)
        correct_index_pos = len(retrieved_indexes) - retrieved_indexes.tolist().index(correct_index)
        if correct_index_pos == 1:
            r1 += 1
        if correct_index_pos <= 5:
            r5 += 1
        if correct_index_pos <= 10:
            r10 += 1
        ranks.append(correct_index_pos)
    median_rank = np.median(ranks)
    return {
        'R@1': r1,
        'R@5': r5,
        'R@10': r10,
        'median_rank': median_rank,
        'R@1_frac': r1/len(ranks),
        'R@5_frac': r5/len(ranks),
        'R@10_frac': r10/len(ranks),
        'median_rank_frac': median_rank/len(ranks)
    }