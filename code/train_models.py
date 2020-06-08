dev = False

## command-line args
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-r', dest='resume', type=int)
parser.add_argument('-e', dest='epochs', type=int, default=10)
parser.add_argument('-g', dest='gpu', type=int, default=0)
parser.add_argument('-b', dest='batch_size', type=int, default=128)
parser.add_argument('-d', dest='development', action='store_true')
parser.add_argument('-t', dest='target_dir', default='saved_models')
parser.add_argument('-l', dest='models_list', default='modes.txt')

args = parser.parse_args()

resume = args.resume
epochs = args.epochs
batch_size = args.batch_size
dev = args.development
dir_path = args.target_dir
models_list = args.models_list

from pathlib import Path

## library
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import numpy as np
import json
from collections import Counter
import gc
from time import time

#from matplotlib import pyplot as plt

## load from files
base_url = 'https://cs.stanford.edu/people/rak248/'

print('loading image profiles')
s = time()
images = {
    item['image_id']: {
        'width'  : item['width'],
        'height' : item['height'],
        'path'   : item['url'].replace(base_url, 'visual_genome/data/')
    }
    for item in json.load(open('visual_genome/data/image_data.json'))
}
e = time()
print('image names loaded {0:0.2f}s'.format(e-s))

# read from file
print('loading relationships corpus')
s = time()
rels_from_file = json.load(open('visual_genome/data/relationships.json'))
e = time()
print('relationships loaded {0:0.2f}s'.format(e-s))

# name/names correction for reading content of nodes in the dataset
def name_extract(x):
    if 'names' in x and len(x['names']):
        name = x['names'][0]
    elif 'name' in x:
        name = x['name']
    else:
        name = ''
    return name.strip().lower()

print('preprocessing relationships with image profiles')
s = time()
# convert it into a set of (image, subject, predicate, object)
triplets_key_values = {
    (
        rels_in_image['image_id'],
        name_extract(rel['subject']),
        rel['predicate'].lower().strip(),
        name_extract(rel['object']),
    ): (
        rels_in_image['image_id'],
        (name_extract(rel['subject']), rel['subject']['object_id'], (rel['subject']['x'],rel['subject']['y'],rel['subject']['w'],rel['subject']['h'])),
        rel['predicate'].lower().strip(),
        (name_extract(rel['object']), rel['object']['object_id'], (rel['object']['x'], rel['object']['y'], rel['object']['w'], rel['object']['h'])),
    )
    for rels_in_image in rels_from_file
    for rel in rels_in_image['relationships']
}
triplets = list(triplets_key_values.values())
del triplets_key_values
e = time()
print('preprocessed relationships {0:0.2f}s'.format(e-s))


print('loading image filters for training')
s = time()
image_ids = list(np.load('visual_genome/data/relationships/image_ids.npy'))

if dev:
    image_ids = image_ids[:batch_size]

# only use the filtered ones
filtered_image_ids = set(list(np.load('visual_genome/data/relationships/image_ids_train.npy')))
e = time()
print('loaded image filters for training {0:0.2f}s'.format(e-s))

print('filtering triplets based on images')
s = time()
triplets = [
    item
    for item in triplets
    #if item[0] in image_ids
    if item[0] in filtered_image_ids
]
gc.collect()

e = time()
print('filtered triplets based on images {0:0.2f}s'.format(e-s))


print('creating filters for bboxes')
s = time()
filtered_obj_ids = set([
    obj_id
    for item in triplets
    for obj_id in [item[1][1], item[3][1]]
])
e = time()
print('created filters for bboxes {0:0.2f}s'.format(e-s))


print('loading images')
chunck_size = 10000
img_visual_features = []
for l in range(0, len(image_ids), chunck_size):
    s = time()
    vfs = np.load('visual_genome/data/relationships/image_resnet50_features_['+str(l)+'].npy', allow_pickle=True)
    img_visual_features += [
        (iid, vf)
        for iid, vf in list(zip(image_ids[l:l+chunck_size], vfs))
        if iid in filtered_image_ids
        if type(vf) != int
    ]
    e = time()
    print('{0} total files are loaded after filtering {1} in {2:0.2f}s'.format(len(img_visual_features), len(vfs), e-s))
    del vfs
    
img_visual_features = dict(img_visual_features)

object_ids = list(np.load('visual_genome/data/relationships/object_ids.npy', allow_pickle=True))

print('loading bboxes')
chunck_size = 100000
visual_features = []
for l in range(0, len(object_ids), chunck_size):
    s = time()
    vfs = np.load('visual_genome/data/relationships/objects_resnet50_features_['+str(l)+'].npy', allow_pickle=True)
    visual_features += [
        (iid, vf)
        for iid, vf in zip(object_ids[l:l+chunck_size], vfs)
        if iid in filtered_obj_ids
        if type(vf) != int
    ]
    e = time()
    print('{0} total files are loaded after filtering {1} in {2:0.2f}s'.format(len(visual_features), len(vfs), e-s))
    del vfs

visual_features = dict(visual_features)


print('removing the triplets with missing pre-processed data')
s = time()
# clean the data from examples in which there is no saved vectors for them!
triplets = [
    item
    for item in triplets
    if item[0] in img_visual_features 
    if type(img_visual_features[item[0]]) != int
    if item[1][1] in visual_features 
    if type(visual_features[item[1][1]]) != int
    if item[3][1] in visual_features 
    if type(visual_features[item[3][1]]) != int
]
e = time()
print('removed the triplets with missing pre-processed data {0:0.2f}s'.format(e-s))

#vocab = Counter([w.strip() for _,(sbj,_,_),pred,(obj,_,_) in triplets for w in ' '.join([sbj,pred,obj]).split(' ')])
#np.save('visual_genome/data/relationships/vocab_caption.npy', vocab)
vocab = np.load('visual_genome/data/relationships/vocab_caption.npy', allow_pickle=True)[None][0]

word2ix = {w:i for i,w in enumerate(['<0>', '<s>']+list(vocab))}
ix2word = {i:w for w,i in word2ix.items()}
word2onehot = lambda w: np.array([0.]*word2ix[w] + [1.] + [0.]*(len(word2ix)-word2ix[w]-1))

max_len = 16 #max(len(' '.join([sbj,pred,obj]).split(' ')) for _,(sbj,_,_),pred,(obj,_,_) in triplets)

np.save('triplets_train.npy', triplets)

print('# vocab_size:', len(vocab))
print('# images:', len(img_visual_features))
print('# bounding boxes:', len(visual_features))
print('# expressions:', len(triplets))


# keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Flatten, AveragePooling2D
from keras.layers import Dense, LSTM, Embedding, Masking
from keras.layers import Input, Lambda, RepeatVector, Reshape, Dropout
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.metrics import sparse_top_k_categorical_accuracy, sparse_categorical_accuracy

from keras.callbacks import EarlyStopping

from keras import backend as K


def item2features(item):
    img_id,(sbj,object_id1,sbj_bbx),pred,(obj,object_id2,obj_bbx) = item

    # visual features
    vf0 = img_visual_features[img_id]
    vf1 = visual_features[object_id1]
    vf2 = visual_features[object_id2]

    # spatial features
    # based on VisKE
    # area of each bbox:
    a1 = sbj_bbx[2] * sbj_bbx[3]
    a2 = obj_bbx[2] * obj_bbx[3]
    # overlap width:
    if obj_bbx[0] <= sbj_bbx[0] <= obj_bbx[0]+obj_bbx[2] <= sbj_bbx[0] + sbj_bbx[2]:
        # overlap
        w = (obj_bbx[0]+obj_bbx[2]) - (sbj_bbx[0])
    elif obj_bbx[0] <= sbj_bbx[0] <= sbj_bbx[0] + sbj_bbx[2] <= obj_bbx[0]+obj_bbx[2]:
        # obj contains sbj
        w = sbj_bbx[2]
    elif sbj_bbx[0] <= obj_bbx[0] <= sbj_bbx[0] + sbj_bbx[2] <= obj_bbx[0]+obj_bbx[2]:
        # overlaps
        w = (sbj_bbx[0]+sbj_bbx[2]) - (obj_bbx[0])
    elif sbj_bbx[0] <= obj_bbx[0] <= obj_bbx[0]+obj_bbx[2] <= sbj_bbx[0] + sbj_bbx[2]:
        # subj contains obj
        w = obj_bbx[2]
    else:
        w = 0

    # overlap height:
    if obj_bbx[1] <= sbj_bbx[1] <= obj_bbx[1]+obj_bbx[3] <= sbj_bbx[1] + sbj_bbx[3]:
        # overlap
        h = (obj_bbx[1]+obj_bbx[3]) - (sbj_bbx[1])
    elif obj_bbx[1] <= sbj_bbx[1] <= sbj_bbx[1] + sbj_bbx[3] <= obj_bbx[1]+obj_bbx[3]:
        # obj contains sbj
        h = sbj_bbx[3]
    elif sbj_bbx[1] <= obj_bbx[1] <= sbj_bbx[1] + sbj_bbx[3] <= obj_bbx[1]+obj_bbx[3]:
        # overlaps
        h = (sbj_bbx[1]+sbj_bbx[3]) - (obj_bbx[1])
    elif sbj_bbx[1] <= obj_bbx[1] <= obj_bbx[1]+obj_bbx[3] <= sbj_bbx[1] + sbj_bbx[3]:
        # subj contains obj
        h = obj_bbx[3]
    else:
        h = 0

    # overlap area
    overlap_a = w * h

    # dx; dy; ov; ov1; ov2; h1;w1; h2;w2; a1; a2
    sf1 = [
        #obj_bbx[0] - sbj_bbx[0], # dx = x2 - x1  #change in corners
        #obj_bbx[1] - sbj_bbx[1], # dy = y2 - y1  #change in corners
        obj_bbx[0] - sbj_bbx[0] + (obj_bbx[2] - sbj_bbx[2])/2, # dx = x2 - x1 + (w2 - w1)/2 #change in centers
        obj_bbx[1] - sbj_bbx[1] + (obj_bbx[3] - sbj_bbx[3])/2, # dy = y2 - y1 + (h2 - h1)/2 #change in centers
        0 if (a1+a2) == 0 else overlap_a/(a1+a2), # ov
        0 if a1 == 0 else overlap_a/a1, # ov1
        0 if a2 == 0 else overlap_a/a2, # ov2
        sbj_bbx[3], # h1
        sbj_bbx[2], # w1
        obj_bbx[3], # h2
        obj_bbx[2], # w2
        a1, # a1
        a2, # a2
    ]    
    
    # spatial template (two attention masks)
    x1, y1, w1, h1 = sbj_bbx
    x2, y2, w2, h2 = obj_bbx

    mask = np.zeros([2,7,7])
    mask[0, int(y1*7):int((y1+h1)*7), int(x1*7):int((x1+w1)*7)] = 1 # mask bbox 1 
    mask[1, int(y2*7):int((y2+h2)*7), int(x2*7):int((x2+w2)*7)] = 1 # mask bbox 2

    sf2 = mask.flatten()
    
    # sentence encoding
    sent = ' '.join([sbj,pred,obj]).split(' ')
    sent = [word2ix['<s>']]+[word2ix[w] for w in sent]+[word2ix['<0>']]*(1+max_len-len(sent))

    return vf0, sf1, sf2, vf1, vf2, sent


# this is memory intensive, it is possible to do this in generator loop as well.
print('preparing data')
e = time()
prepared_data = [
    item2features(item)
    for item in triplets
]
e = time()
del visual_features
del img_visual_features
del triplets
gc.collect()
print('prepared data {0:0.2f}s'.format(e-s))


def reverse_viske(sf):
    return [-sf[0], -sf[1], sf[2], sf[4], sf[3],  sf[7], sf[8], sf[5], sf[6], sf[10], sf[9], ]

def reverse_mask(sf):
    return np.concatenate([sf[49:], sf[:49]])


def generator_features_description(batch_size=32, split=(0.,1.), all_data = prepared_data, mode='bbox'):
    while True:
        gc.collect()
        
        # shuffle 
        _all_data = all_data[int(len(all_data)*split[0]):int(len(all_data)*split[1])]
        np.random.shuffle(_all_data)
        
        # start
        X_vfs = []
        X_sfs = []
        X_objs = []
        X_sents = []
        
        for item in _all_data:
            #vf0, sf1, sf2, vf1, vf2, sent = item2features(item)
            vf0, sf1, sf2, vf1, vf2, sent = item
            
            # add to the batch
            X_vfs.append(vf0)
            X_sents.append(sent)
            
            # list of objects 
            l = [vf1, vf2]
            # if it needs to be shuffled
            if mode[-2:] == '-r':
                #np.random.shuffle(l)
                if np.random.random() > 0.5:
                    l = [vf2, vf1]
                    sf1 = reverse_viske(sf1)
                    sf2 = reverse_mask(sf2)

            # two types of spatial features:
            if mode[:4] == 'bbox' or mode[-4:] == 'bbox':
                X_sfs.append(sf1)
            elif mode[:9] == 'attention':
                X_sfs.append(sf2)
            
            # two visual features from two bounding boxes 
            if mode[:4] == 'bbox' or mode[:9] == 'attention' or mode[:8] == 'implicit':                        
                X_objs.append(l)

            # add to flush the batch if needed
            if len(X_sents) == batch_size:
                sents = np.array(X_sents)
                if mode[:4] == 'bbox' or mode[:9] == 'attention':
                    yield ([np.array(X_vfs), np.array(X_sfs), np.array(X_objs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))
                    
                elif mode[:8] == 'implicit':
                    yield ([np.array(X_vfs), np.array(X_objs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))
                    
                elif mode == 'spatial_adaptive-bbox':
                    yield ([np.array(X_vfs), np.array(X_sfs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))

                elif mode[:7] == 'no-beta' or mode == 'spatial_adaptive' or mode == 'spatial_adaptive-attention':
                    yield ([np.array(X_vfs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))
                    
                X_vfs = []
                X_sfs = []
                X_objs = []
                X_sents = []


def build_model(mode='bbox'):
    print('mode:', mode)
    
    unit_size = 200
    dropout_rate = 0.5
    regions_size = 7 * 7
    beta_size = 2 + 1 + 1 # 2 objects + 1 sentential + 1 spatial

    delayed_sent = Input(shape=[max_len+1])
    
    sf_size = 11 # dx; dy; ov; ov1; ov2; h1; w1; h2; w2; a1; a2 (from VisKE)
    beta_feature_size = 2*(beta_size-1)*unit_size
    
    if mode[:9] == 'attention':
        sf_size = 49*2  # attention mask pattern
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode[:4] == 'bbox':
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode[:8] == 'implicit':
        beta_size = 2 + 1
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode == 'spatial_adaptive':
        beta_size = regions_size + 1
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode == 'spatial_adaptive-bbox':
        beta_size = regions_size + 1 + 1
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode == 'spatial_adaptive-attention':
        sf_size = 49
        beta_size = regions_size + 1 + 1
        beta_feature_size = 2*(beta_size-2)*unit_size
        
    visual_features_in0 = Input(shape=[regions_size, 2048]) # resnet50
    visual_features_objs_in = Input(shape=[2, 2048]) # resnet50
    spatial_features_in = Input(shape=[sf_size]) 

    embeddings = Embedding(len(word2ix), unit_size)(Masking()(delayed_sent))
    
    # fine tune / project features
    def mlp_vision(x): 
        out = TimeDistributed(Dense(unit_size, activation='tanh'))(x)
        out = TimeDistributed(Dense(unit_size, activation='relu'))(out)
        out = Dropout(dropout_rate)(out)
        return out

    # geometric spatial features
    def mlp_space(x):
        out = Dense(unit_size, activation='tanh')(x)
        out = Dense(unit_size, activation='relu')(out)
        out = Dropout(dropout_rate)(out)
        return out

    # attention alpha
    def mlp_att(x):
        out = TimeDistributed(Dense(unit_size, activation='relu'))(x)
        out = Dropout(dropout_rate)(out)
        out = TimeDistributed(Dense(unit_size, activation='tanh'))(out)
        out = TimeDistributed(Dense(beta_size, activation='softmax'))(out)
        return out
    
    ### global visual features
    visual_features0 = mlp_vision(visual_features_in0) # learn to find
    visual_features0_g = Reshape([7,7,unit_size])(visual_features0)
    visual_features0_g = Flatten()(AveragePooling2D([7,7])(visual_features0_g))
    #visual_features0_g = Reshape([7,7,2024])(visual_features_in0)
    #visual_features0_g = mlp_vision(AveragePooling2D([7,7])(visual_features0_g))
    
    ### objects visual features (top-down bboxes)
    visual_features_objs = mlp_vision(visual_features_objs_in)
    
    ### geometric spatial features (bbox geometry)
    spatial_features = mlp_space(spatial_features_in)

    ### adaptive attention: beta (which feature set needs more attention?)
    def feature_fusion(x, regions_size=regions_size, max_len=max_len):
        return K.concatenate([
            x[0],
            K.repeat_elements(K.expand_dims(x[1], 1), max_len+1, 1),
        ], 2)
    
    ## concatenate features for attention model
    def beta_features(x, unit_size=unit_size, max_len=max_len, beta_size=beta_size):
        if mode[:8] == 'implicit' or mode == 'spatial_adaptive' or mode=='spatial_adaptive-attention':
            h, vf0 = x
            #h = K.l2_normalize(h, -1)
            #vf0 = K.l2_normalize(vf0, -1)

            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1) # [sent, 49, unit_size] or [sent, 2, unit_size]
            if mode=='spatial_adaptive-attention':
                h_   = K.repeat_elements(K.expand_dims(h, 2), beta_size-2, 2) # [sent, 49, unit_size]
                return K.reshape(K.concatenate([h_, vf0_], 3), [-1, max_len+1, 2*(beta_size-2)*unit_size]) # [sent, 49*b*unit_size]
            else:
                h_   = K.repeat_elements(K.expand_dims(h, 2), beta_size-1, 2) # [sent, 49, unit_size]
                return K.reshape(K.concatenate([h_, vf0_], 3), [-1, max_len+1, 2*(beta_size-1)*unit_size]) # [sent, 49*b*unit_size]
        else:
            h, sf, vf0 = x
            #h = K.l2_normalize(h, -1)
            #sf = K.l2_normalize(sf, -1)
            #vf0 = K.l2_normalize(vf0, -1)

            sf_  = K.expand_dims(K.repeat_elements(K.expand_dims(sf, 1), max_len+1, 1), 2) # [sent, 1, unit_size]
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1) # [sent, 49, unit_size] or [sent, 2, unit_size]
            vf_sf = K.concatenate([sf_,vf0_],2) # [sent, 49+1, unit_size] or [sent, 2+1, unit_size]
            
            h_   = K.repeat_elements(K.expand_dims(h, 2), beta_size-1, 2) # [sent, 49+1, unit_size] or or [sent, 2+1, unit_size]

            return K.reshape(K.concatenate([h_, vf_sf], 3), [-1, max_len+1, 2*(beta_size-1)*unit_size]) # [sent, 49+1*b*unit_size]

    
    ### apply adaptive attention (beta)
    def adaptation_attention(x, max_len=max_len, regions_size=regions_size, mode=mode):
        if mode[:8] == 'implicit' or mode == 'spatial_adaptive':
            h, vf0, b = x
            
            # normalize them as they are comparable in the same vector space
            vf0 = K.l2_normalize(vf0, -1)
            h = K.l2_normalize(h, -1)
            
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1)
            
            return b[:, :, 0:1] * h + K.sum(K.expand_dims(b[:, :, 1:], 3) * vf0_, 2)
        else:
            h, sf, vf0, b = x
            
            # normalize them as they are comparable in the same vector space
            vf0 = K.l2_normalize(vf0, -1)
            sf = K.l2_normalize(sf, -1)
            h = K.l2_normalize(h, -1)

            if len(sf.get_shape()) == 2:
                sf_ = K.repeat_elements(K.expand_dims(sf, 1), max_len+1, 1)
            else:
                sf_ = sf
               
            
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1)
            
            return b[:, :, 0:1] * h + b[:, :, 1:2] * sf_ + K.sum(K.expand_dims(b[:, :, 2:], 3) * vf0_, 2)
    
    fused_features = Lambda(feature_fusion)([embeddings, visual_features0_g])
    hidden_a       = LSTM(unit_size, return_sequences=True, dropout=dropout_rate,)(fused_features)
    ling_features  = LSTM(unit_size, return_sequences=True, dropout=dropout_rate,)(hidden_a)

    if mode[:4] == 'bbox' or mode[:9] == 'attention':
        beta_features_out = Lambda(beta_features)([hidden_a, spatial_features, visual_features_objs])
        beta = mlp_att(beta_features_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, spatial_features, visual_features_objs, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, spatial_features_in, visual_features_objs_in, delayed_sent], out)
    elif mode[:8] == 'implicit':
        beta_features_out = Lambda(beta_features)([hidden_a, visual_features_objs])
        beta = mlp_att(beta_features_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, visual_features_objs, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, visual_features_objs_in, delayed_sent], out)
    elif mode[:7] == 'no-beta':
        out = Dense(len(word2ix), activation='softmax')(fused_features)
        model = Model([visual_features_in0, delayed_sent], out)
    elif mode == 'spatial_adaptive':
        beta_features_out = Lambda(beta_features)([hidden_a, visual_features0])
        beta = mlp_att(beta_features_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, visual_features0, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, delayed_sent], out)
    elif mode == 'spatial_adaptive-bbox':
        beta_features_out = Lambda(beta_features)([hidden_a, spatial_features, visual_features0])
        beta = mlp_att(beta_features_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, spatial_features, visual_features0, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, spatial_features_in, delayed_sent], out)
    elif mode == 'spatial_adaptive-attention':
        beta_features_out = Lambda(beta_features)([hidden_a, visual_features0])
        beta = mlp_att(beta_features_out)
        beta_spatial = Lambda(lambda x: x[:, :, 2:])(beta)
        spatial_features = mlp_space(beta_spatial)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, spatial_features, visual_features0, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, delayed_sent], out)

    model.summary()
    model.compile(
        optimizer=Adam(lr=5e-4, beta_1=0.9, beta_2=0.995),
        loss='sparse_categorical_crossentropy',
        #metrics=[sparse_top_k_categorical_accuracy, sparse_categorical_accuracy],
    )

    return model


if dev:
    dir_path = '{}/dev_'.fotmat(dir_path)
    
modes = [
    mode.strip()
    for mode in open(models_list, 'r')
    if len(mode) > 0
]

def train(model, sch=-1):
    batch_schedule = [
        (batch_size * (2**i), epochs) # double the batch size on each step
        for i in range(5)
    ]
    
    for i, (bs, ne) in enumerate(batch_schedule):
        if i <= sch:
            continue
        h = model.fit_generator(
            generator=generator_features_description(batch_size=bs, split=(0.,0.95), mode=mode), 
            steps_per_epoch=int(len(prepared_data)*0.95/bs) if not dev else 10, 
            validation_data=generator_features_description(batch_size=bs, split=(0.95,1.), mode=mode),
            validation_steps=int(len(prepared_data)*0.05/bs) if not dev else 10,
            epochs=ne,
            callbacks=[EarlyStopping(patience=0)], # zero patience
        )

        # updated version new file names!!
        model.save(dir_path + '/caption_model_{0}_{3}sch_{1}e_{2}bs.h5'.format(mode, len(h.history['val_loss']), bs, i))
        np.save(dir_path + '/caption_model_{0}_{3}sch_{1}e_{2}bs_history.npy'.format(mode, len(h.history['val_loss']), bs, i), h.history)


for mode in modes:
    saved_schedules = sorted([
        (int(re.findall('.+(\d)sch.+', filename)[0]), filename)
        for filename in glob(dir_path + '/caption_model_{0}_*.h5'.format(mode)):
        if len(re.findall('.+(\d)sch.+', filename)) == 1
    ])
    
    if not resume or len(saved_schedule) != 1:
        # start from scrach:
        model = build_model(mode=mode)
        train(model)
    else:
        sch, filename = saved_schedules[-1]
        model = load_model(filename)
        train(model, sch)

