import gensim
from gensim.models import Word2Vec
import pandas as pd
import re

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

response = []
# Import train_rel_2.tsv into Python
with open('resources/СборникСтатейДляОбучения.tsv', encoding='utf-8') as f:
    lines = f.readlines()
    # print(lines)
    columns = lines[0].split('\t')
    for line in lines[1:]:
        temp = line.lower().split('\t')
        if len(temp) <= 2: continue; 
        # if temp[1] == '2':   # Select the Essay Set 2
        response.append(re.sub(patterns, ' ', temp[0]))  # Select "EssayText" as a corpus

data = pd.DataFrame(list(zip(response)))
data.columns = ['response']
response_base = data.response.apply(gensim.utils.simple_preprocess, min_len=4)

_min_count = 2
_negative = 10
model = Word2Vec(
    sentences=response_base,
    min_count=_min_count,
    window=2,
    vector_size=64,
    alpha=0.03,
    negative=_negative,
    min_alpha=0.0007,
    sample=6e-5
)

print(f"{len(model.wv)=}")
print(f"{model.wv['сапр']=}")
print(f"{model.wv['анализ']=}")
print(f"{model.wv['кластеризация']=}")
print(f"{model.wv['проектирование']=}")
print(f"{model.corpus_count=}")

# Train the model
model.build_vocab(response_base, update=True)
model.train(response_base, total_examples=model.corpus_count, epochs=model.epochs)

print(f"{len(model.wv)=}")
print(f"{model.wv['сапр']=}")
print(f"{model.corpus_count=}")
print(f"{model.wv.most_similar('сапр')=}")
print(f"{model.wv.most_similar('анализ')=}")
print(f"{model.wv.most_similar('кластеризация')=}")
print(f"{model.wv.most_similar('проектирование')=}")
print(f"{model.wv.most_similar_to_given('сапр', ['система', 'программа'])=}")

model.save(f"resources/w2v_sapr_min_count{_min_count}_wv{len(model.wv)}_negative{_negative}.model")