import os
import sklearn
from fastai.text import *
import html
import sys
import numpy as np
import torch
import torch.tensor as T
import argparse
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

class ClaimMiner(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.learner = load_learner(self.model_path)

    def load_lm_data(self, data_path):
        texts = []
        for line in open(data_path, encoding='utf-8'):
            texts.append(line.strip())
        
        col_names = ['labels','text']
        trn_texts, val_texts = train_test_split(texts, test_size=0.1)

        df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
        df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

        df_trn['text'] = df_trn['text'].apply(lambda x: fixup(x))
        df_val['text'] = df_val['text'].apply(lambda x: fixup(x))

        return df_trn, df_val

    def load_classifier_data(self, data_path):
        texts  = []
        labels = [] 
        for line in open(data_path, encoding='utf-8'):

            text_label = line.strip().split('\t')
            if len(text_label) > 1:
                texts.append(text_label[0])
                labels.append(int(text_label[1]))
        
        df = pd.DataFrame({'text':texts, 'labels':labels}, columns=['labels','text'])
        df['text'] = df['text'].apply(lambda x: fixup(x))
        
        df_trn, df_val = train_test_split(df, test_size=0.1)
        
        
        return df_trn, df_val


    def train_lm(self, data_path):
        df_trn, df_val = load_lm_data(data_path)

        LM_PATH=Path(self.model_path)
        LM_PATH.mkdir(exist_ok=True)

        df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
        df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)

        data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, 
            path = self.model_path)

        #Train a language model..
        learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.7)
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 1e-7)
        learn.save_encoder('ft_enc')

    def train_classifier(self, data_path):
        trn_df, val_df = load_classifier_data(data_path)
        
        LM_PATH=Path(output_path + '/argument_lm/')
        LM_PATH.mkdir(exist_ok=True)

        data_clas = TextClasDataBunch.from_df(path = self.model_path, train_df = trn_df, 
            valid_df = val_df, vocab=data_lm.train_ds.vocab, bs=32)

        learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.7)
        learn.load_encoder('ft_enc')

        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 1e-3)

        learn.save('lm_last_ft')
        learn.load('lm_last_ft')

        learn.unfreeze()
        learn.fit_one_cycle(10, 1e-3)

        learn.export()

    def predict(self, texts):
        if self.learner == None:
            self.learner = load_learner(self.model_path)

        preds = [self.learner.predict(text)[2] for text in texts]

        return preds



def get_claim(cminer, text, topic):
    sents = sent_tokenize(text)
    #filter only sentences that overlap with the topic...
    sents_preds   = [cminer.learner.predict(sent) for sent in sents]
    
    sents_out = [(x[0], (x[1][2][0].item(), x[1][2][1].item())) for x in zip(sents, sents_preds)]
    sents_out = sorted(sents_out, key=lambda x: -x[1][1])
    
    return sents_out

def choose_claim(claims_scores, min_len=15):
    if len(claims_scores) > 0:
        #filter out short sentences
        claims_scores = [x for x in claims_scores if len(x[0].split(' ')) > min_len]
        if len(claims_scores) == 0:
            return ''
        else:
            return sorted(claims_scores, key=lambda x: -x[1][1])[0][0]
    else:
        return ''

def mine_claim_from_df(df_path, cminer_path, output_path):
    #'/workspace/ceph_data/belief-based-argumentation-generation/models/arg_mining'
    cminer = ClaimMiner(cminer_path)
    df = pd.read_csv(df_path)

    df['claims'] = df.apply(lambda row: get_claim(cminer, row['opinion_txt'], row['topic']), axis=1)
    df['top_claim'] = df.claims.apply(lambda x: choose_claim(x))

    df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument mining')
    parser.add_argument('--data_frame_path')
    parser.add_argument('--output_path')
    parser.add_argument('--cminer_path')

    args = parser.parse_args()

    mine_claim_from_df(args.data_frame_path, args.cminer_path, args.output_path)

