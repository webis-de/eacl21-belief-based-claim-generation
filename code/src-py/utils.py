import nltk
import json
import numpy as np
import pickle
import pandas as pd

from nltk import sent_tokenize, word_tokenize
from nltk import RegexpParser, ngrams, pos_tag
from nltk.corpus import stopwords, wordnet

import spacy

nlp = spacy.load("en")

ideology_map = {
          'Anarchist': 'left',
         'Communist': 'left',
         'Labor': 'left',
         'Progressive': 'left',
         'Socialist' : 'left',
         'Conservative': 'right',
         'Liberal': 'left',
         'Libertarian': 'other',
         'Apathetic': 'other',
         'Green': 'left',
         'Moderate': 'other',
         'Not Saying': 'unkown',
         'Other': 'unkown',
         'Undecided': 'unkown'
}

issue_map = {
    'Con': 1,
    'N/S': 0,
    'N/O': 2,
    'Und': 0,
    'Pro': 3,
}


big_issues = ['Abortion',
 'Affirmative Action',
 'Animal Rights',
 'Barack Obama',
 'Border Fence',
 'Capitalism',
 'Civil Unions',
 'Death Penalty',
 'Drug Legalization',
 'Electoral College',
 'Environmental Protection',
 'Estate Tax',
 'European Union',
 'Euthanasia',
 'Federal Reserve',
 'Flat Tax',
 'Free Trade',
 'Gay Marriage',
 'Global Warming Exists',
 'Globalization',
 'Gold Standard',
 'Gun Rights',
 'Homeschooling',
 'Internet Censorship',
 'Iran-Iraq War',
 'Labor Union',
 'Legalized Prostitution',
 'Medicaid & Medicare',
 'Medical Marijuana',
 'Military Intervention',
 'Minimum Wage',
 'National Health Care',
 'National Retail Sales Tax',
 'Occupy Movement',
 'Progressive Tax',
 'Racial Profiling',
 'Redistribution',
 'Smoking Ban',
 'Social Programs',
 'Social Security',
 'Socialism',
 'Stimulus Spending',
 'Term Limits',
 'Torture',
 'United Nations',
 'War in Afghanistan',
 'War on Terror',
 'Welfare']

def get_longest_np(text):
        doc = nlp(text)
        
        nps = [str(x) for x in doc.noun_chunks]
        
        if len(nps) == 0:
            return text

        return sorted(nps, key=lambda x: - len(x))[0]

def extract_keyphrases(sentence: str) -> list:
    STOP_WORDS = set(stopwords.words('english'))
    
    keyphrases = []
    words = word_tokenize(sentence)
    tagged = pos_tag(words)

    grammar = '''
        NP: {<DT|PP$>?<JJ|JJR>*<NN.*|CD|JJ>+}
        PP: {<IN><NP>}
        VP: {<MD>?<VB.*><NP|PP>}
        '''
    chunked = RegexpParser(grammar)
    tree = chunked.parse(tagged)
    for subtree in tree.subtrees():
        if subtree.label() == 'NP' or subtree.label() == 'VP':
            keyphrase = [word for word,
                         tag in subtree.leaves() if not word in STOP_WORDS]
            if 1 <= len(keyphrase) <= 10:
                keyphrases.append(' '.join(keyphrase).lower())

    return keyphrases

def overlap_perc(key_phrases, argument):
    if len(key_phrases) == 0:
        return 0
    
    overlap = 0
    for key_phrase in key_phrases:
        if key_phrase.lower() in argument.lower():
            overlap +=1
    
    return overlap/len(key_phrases)
    
def visualize_graph(model):
    import torch
    from torch import nn
    from torchviz import make_dot, make_dot_from_trace

    x = app_translator.model(torch.ones(3, 8,1).type(torch.LongTensor).cuda(), 
                             torch.ones(3, 8,1).type(torch.LongTensor).cuda(), torch.ones(8).cuda(), 
                             torch.ones(8, 16).cuda(), torch.ones(8, 5, 300).cuda(), torch.ones(8).cuda())
    g = make_dot(x[0])
    g.render(filename='./model_graph', format='pdf')

def get_sorted_claims(cminer, text):
    sents = sent_tokenize(text)
    sents_preds = [cminer.learner.predict(sent) for sent in sents]
    sents_preds = [(x[0], (x[1][2][0].item(), x[1][2][1].item())) for x in zip(sents, sents_preds)]
    
    return sorted(sents_preds, key=lambda x: -x[1][1])

def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


def encode_phrase(phrase, encoding_model):
    tokens =[token.lower() for token in phrase.split(' ')]
    token_vecs = []
    for token in tokens:
        if token in encoding_model:
            token_vecs.append(encoding_model[token])

    if len(token_vecs) == 0:
        print('Empty phrase.. ')
        return None #np.random.rand(300)
    else:
        return np.mean(token_vecs, axis=0)

def get_topic_key_phrases(topic_key_phrases_dict, topic):
    if topic in topic_key_phrases_dict:
        all_aspects = [x['aspects'] for x in topic_key_phrases_dict[topic].values() if type(x) is not list]
        return set([item for argument_aspects in all_aspects for item in argument_aspects])
    else:
        return set()

def desc_user_from_vec(vector):
    desc = {}
    for vec_idx, value in enumerate(vector):
        if value in [1, 3]:
            desc[big_issues[vec_idx]] = value

    return desc


def build_user_bi_vec(feats_values):
    user_beliefs = {x:0 for x in big_issues}
    for key, value in feats_values.items():
        user_beliefs[key] = issue_map[value]
    user_vector = np.array(list(x[1] for x in user_beliefs.items()))

    return user_vector


def get_topic_encoded_key_phrases(topic_key_phrases_dict, encoding_model, topic):
    if topic in topic_key_phrases_dict:
        all_aspects = [x['aspects'] for x in topic_key_phrases_dict[topic].values() if type(x) is not list]
        aspects     = set([item for argument_aspects in all_aspects for item in argument_aspects])

        #embed aspects
        aspects_vectors = [encode_phrase(x, encoding_model) for x in aspects]
        return aspects_vectors
    else:
        return []