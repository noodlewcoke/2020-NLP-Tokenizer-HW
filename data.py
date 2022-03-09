import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np 
import json

data_path = 'data/'
FILENAMES = [
            'en.wiki.sentences.train',
            'en.wiki.sentences.dev',
            'en.wiki.merged.sentences.train',
            'en.wiki.merged.sentences.dev'
            ]
GOLDNAMES = [
            'en.wiki.gold.train',
            'en.wiki.gold.dev',
            'en.wiki.merged.gold.train',
            'en.wiki.merged.gold.dev'
            ]
TESTNAMES = [
            'en.wiki.merged.sentences.test',
            'en.wiki.sentences.test'
            ]
outputNames = [
                'en.wiki.train',
                'en.wiki.dev',
                'en.wiki.merged.train',
                'en.wiki.merged.dev',
                'en.wiki.merged.test',
                'en.wiki.test'
]

with open('vocabulary.json', 'r') as f:
    vocab = json.load(f)
    f.close()
vocabulary = vocab['dict']

# Construct dataset for each file.
# '<PAD>' -> 0, 'B' -> 1, 'I' -> 2, 'S' -> 3

print("Constructing train and dev sets.")

c = 0
for s, g in zip(FILENAMES, GOLDNAMES):
    with open(data_path+s, 'r') as f:
        sentences = f.read().split('\n')
        f.close()
    with open(data_path+g, 'r') as f:
        labels = f.read().split('\n')
        f.close()
    dataset = []
    for sentence, label in zip(sentences, labels):
        if not sentence:
            continue
        assert len(sentence) == len(label), "A sentence-label pair doesn't hold."
        enumSentence = [vocabulary[s] for s in sentence]
        enumLabels = [] 
        for l in label:
            if l == 'B':
                enumLabels.append(1)
            elif l == 'I':
                enumLabels.append(2)
            elif l == 'S':
                enumLabels.append(3)
        dataset.append((enumSentence, enumLabels))
    
    np.save(outputNames[c], dataset)
    c += 1

print("Constructing test sets.")
unkCount = 0
for t in TESTNAMES:
    with open(data_path+t, 'r') as f:
        sentences = f.read().split('\n')
        f.close()
    dataset = []
    for sentence in sentences:
        if not sentence:
            continue
        enumSentence = [] 
        for s in sentence:
            if s in vocabulary.keys():
                enumSentence.append(vocabulary[s])
            else:
                enumSentence.append(vocabulary['<UNK>'])
                unkCount += 1
        dataset.append(enumSentence)
    np.save(outputNames[c], dataset)
    c += 1

print('DONE. Unknown symbol count: ', unkCount)