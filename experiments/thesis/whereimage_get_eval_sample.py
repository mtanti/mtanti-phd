import random
import PIL.Image

from framework import lib
from framework import data
from framework import config

sample_size = 200
full_img_max_side = 500
thumb_img_max_side = 100

lib.create_dir(config.results_dir+'/whereimage')
lib.create_dir(config.results_dir+'/whereimage/_sample')

for dataset_name in ['mscoco']:
    print(dataset_name)
    
    lib.create_dir(config.results_dir+'/whereimage/_sample/'+dataset_name)
    lib.create_dir(config.results_dir+'/whereimage/_sample/'+dataset_name+'/full')
    lib.create_dir(config.results_dir+'/whereimage/_sample/'+dataset_name+'/thumb')
    
    datasources = data.load_datasources(dataset_name)
    
    images = datasources['test'].get_filenames()
    
    caps = dict()
    caps['human'] = [ [ ' '.join(sent) for sent in group ] for group in datasources['test'].tokenize_sents().get_text_sent_groups() ]
    for architecture in ['init', 'pre', 'par', 'merge']:
        caps[architecture] = [ list() for _ in range(len(images)) ]
        for run in range(1, config.num_runs+1):
            dir_name = '{}_{}_{}'.format(architecture, dataset_name, run)
            with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/shuffled_test_indexes.txt', 'r', encoding='utf-8') as f:
                shuffled_indexes_map = { int(linenum): int(index) for (linenum, index) in enumerate(f.read().strip().split('\n')) }
            with open(config.results_dir+'/whereimage/'+architecture+'/'+dir_name+'/sents.txt', 'r', encoding='utf-8') as f:
                for (i, line) in enumerate(f.read().strip().split('\n')):
                    caps[architecture][shuffled_indexes_map[i]].append(line)
    
    with open(config.results_dir+'/whereimage/_sample/'+dataset_name+'/data.sql', 'w', encoding='utf-8') as f:
        print('INSERT INTO `tbl_data`(`image`, `cap_human`, `cap_init`, `cap_pre`, `cap_par`, `cap_merge`) VALUES', file=f)
        max_str_len = 0
        rand = random.Random(0)
        indexes = list(range(len(images)))
        rand.shuffle(indexes)
        indexes = indexes[:sample_size]
        for index in indexes:
            row = [images[index]] + [ rand.choice(caps[architecture][index]) for architecture in ['human', 'init', 'pre', 'par', 'merge'] ]
            print('(', file=f)
            for field in row:
                if len(field) > max_str_len:
                    max_str_len = len(field)
                print('\t', '\''+field.replace('\'', '\\\'')+'\'', end='', file=f)
                if field != row[-1]:
                    print(',', file=f)
                else:
                    print('', file=f)
            if index != indexes[-1]:
                print('),', file=f)
            else:
                print(');', file=f)
            
            img = PIL.Image.open(config.img_dir(dataset_name)+'/'+images[index])
            (img_width, img_height) = img.size
            
            scale_full = full_img_max_side/max(img_width, img_height)
            img_full = img.resize((round(img_width*scale_full), round(img_height*scale_full)), PIL.Image.BICUBIC)
            img_full.save(config.results_dir+'/whereimage/_sample/'+dataset_name+'/full/'+images[index])
            
            scale_thumb = thumb_img_max_side/max(img_width, img_height)
            img_thumb = img.resize((round(img_width*scale_thumb), round(img_height*scale_thumb)), PIL.Image.BICUBIC)
            img_thumb.save(config.results_dir+'/whereimage/_sample/'+dataset_name+'/thumb/'+images[index])
            
    print(' max cap len:', max_str_len)