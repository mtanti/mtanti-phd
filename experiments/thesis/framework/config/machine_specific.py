base_dir = '' #enter directory where mtanti-phd folder is stored

def data_dir(dataset_name):
    if dataset_name == 'lm1b':
        return base_dir + '/datasets/text/lm1b/1-billion-word-language-modeling-benchmark-master' #Google 1B corpus raw data expected
    else:
        return base_dir + '/datasets/capgen/'+dataset_name+'/captions' #Karpathy raw data expected
        
def img_dir(dataset_name):
    if dataset_name == 'flickr8k':
        return base_dir + '/datasets/capgen/flickr8k/images' #Flickr8k images expected
    elif dataset_name == 'flickr30k':
        return base_dir + '/datasets/capgen/flickr30k/images' #Flickr30k images expected
    elif dataset_name == 'mscoco':
        return base_dir + '/datasets/capgen/mscoco/images' #MSCOCO images expected

mscoco_eval_dir = base_dir + '/tools/coco-caption-master' #MSCOCO evaluation toolkit expected

vgg16_dir = base_dir + '/tools/vgg16' #VGG16 CNN expected

val_batch_size = 100