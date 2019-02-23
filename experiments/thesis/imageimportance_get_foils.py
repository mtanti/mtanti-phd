import nltk
from scipy.spatial import distance

from framework import lib
from framework import data
from framework import config

lib.create_dir(config.results_dir+'/imageimportance')

for dataset_name in ['flickr8k', 'flickr30k', 'mscoco']:
    print(dataset_name)
    datasources = data.load_datasources(dataset_name)
    datasources['test'].tokenize_sents()
    
    image_keywords = [
            {
                token
                for sent in sent_group
                for (token, tag) in nltk.pos_tag(sent, tagset='universal')
                if tag == 'NOUN'
            }
            for sent_group in datasources['test'].get_text_sent_groups()
        ]
    
    prog = lib.ProgressBar(len(image_keywords), 5)
    with open(config.results_dir+'/imageimportance/foils_'+dataset_name+'.txt', 'w', encoding='utf-8') as f:
        for (i, (curr_img, curr_keywords)) in enumerate(zip(datasources['test'].images, image_keywords)):
            index = min(
                range(len(image_keywords)),
                key=lambda j:(image_keywords[j] & curr_keywords, -distance.cosine(datasources['test'].images[j], curr_img))
            )
            print(index, file=f)
            prog.update_value(i+1)
    print()
    print()