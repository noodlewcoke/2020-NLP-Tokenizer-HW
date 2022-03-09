import os  
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy
import json 

#? Create a vocabulary
##* Vocabulary consists of unique symbols in the text. Punctuation symbols also included.
##* For that I'll process all train and dev files.
##* First element of the vocabulary is the <PAD> tag, the last one is the <UNK> tag.

FNAMES = ['en.wiki.sentences.train',
            'en.wiki.sentences.dev',
            'en.wiki.merged.sentences.train',
            'en.wiki.merged.sentences.dev'
        ]

vocabulary = []
for FNAME in FNAMES:
    with open('data/'+FNAME, 'r') as f:
        data = f.read().split('\n')
        f.close()
        textdata = ''.join(data)
        unique = list(set(textdata))
        vocabulary += unique

vocabulary = sorted(list(set(vocabulary)))
vocabulary.insert(0, '<PAD>')
vocabulary.append('<UNK>')
print(len(vocabulary))
vocabulary_dict = {k:v for v, k in enumerate(vocabulary)}
print(vocabulary_dict)

save_dict = {
            'list' : vocabulary,
            'dict' : vocabulary_dict
}

f = open('vocabulary.json', 'w')
json.dump(save_dict, f)
f.close()

