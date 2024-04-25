import re
import os
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

dir_path = 'resources'
w2v_model = Word2Vec.load(dir_path + "/w2v_sapr_min_count2_wv4252_negative10.model")

means = {}
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
for i in list(os.listdir(dir_path)):
    words = 1
    results = 0
    file_name = f"{dir_path}/{i}"
    if '.txt' not in file_name: continue; 
    with open(file_name, 'r', encoding='utf-8') as file:
        print(file_name)
        for line in file:
            line = re.sub(patterns, ' ', line)
            for word in line.split():
                if w2v_model.wv.has_index_for(word) and len(word) > 3:
                    results+=w2v_model.wv.get_vector(word).sum()
                    words+=1
    means[file_name] = results / words
w2v_model.wv.sort_by_descending_frequency()

r = []
for key in means:
    r.append(means[key])
r = sorted(r)

vectors = list(split(r,3))

vec_mean = []
for i in vectors:
    sum = 0
    count = 0
    for vec in i:
        sum+=vec
        count+=1
    vec_mean.append(sum/count)

for i in vec_mean:
    print(i)
    print(w2v_model.wv.similar_by_vector(np.array(i), topn=7, restrict_vocab=None))
    # print(w2v_model.wv.most_similar(positive=[np.array(i)], topn=7))

