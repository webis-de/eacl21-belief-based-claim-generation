import sys

sys.path.insert(0, "/workspace/believe-based-argumentation-generation/src-py/")

import os
import pickle
import pandas as pd
import numpy as np
import random
import codecs
import nltk
import joblib
import itertools
import json
import argparse

from tabulate import tabulate

from opennmt_generation import *
from evaluation import *
from opennmt_data_preparation import get_keyphrases_sentences

import utils

device = 'cuda:0'


data_path = '/workspace/ceph_data/belief-based-argumentation-generation/experiments/experiment_8/'
config_path = '/workspace/ceph_data/belief-based-argumentation-generation/config/opt_config.pickle'
models_path = '/workspace/ceph_data/belief-based-argumentation-generation/models'


def gen_preds(valid_df_path, train_df_path, user_info_path,
     seq2seq_basic_model_path,
     seq2seq_app_model_path,
     output_path,
     topic_key_phrases_path=None,
     min_length=35,
     max_length=100,
     beam_size=10,
     simple_context=True):
    

    train_df= pd.read_csv(train_df_path)
    dev_df  = pd.read_csv(valid_df_path)


    #Filter only users in the training data..
    training_users = set(train_df.user.tolist())
    dev_df  = dev_df[dev_df.user.isin(training_users)]


    #Load extra user information
    users_info = pd.read_parquet(user_info_path)
    user_to_big_issue_dict = pd.Series(users_info.big_issues.values, index=users_info.user).to_dict()
    user_to_embedding_dict = pd.Series(users_info.autoencoder.values, index=users_info.user).to_dict()
    #Load extra user info like; embeddings, big_issues
    dev_df['user_embedding'] = dev_df.user.apply(lambda x: user_to_embedding_dict[x])
    dev_df['big_issues']     = dev_df.user.apply(lambda x: user_to_big_issue_dict[x])
    dev_df['user_belief']    = dev_df.user_ideology.apply(lambda x: utils.ideology_map[x])



    #Start predicting
    ideologies   = dev_df['user_ideology'].tolist()
    user_beliefs = dev_df['user_belief'].tolist()
    topics       = [ '<topic> ' + x for x in dev_df['topic']]
    #topics_and_sentences = [ '<topic> ' + x[0] + ' <sents> ' + ' <sent> '.join(x[1]) for x in zip(dev_df['topic'], dev_df['topic_keysentences'])]
    users        = dev_df['user'].tolist()
    users_big_issues  = np.array(dev_df['big_issues'].tolist()).astype(np.float32)
    users_embedding   = np.array(dev_df['user_embedding'].tolist()).astype(np.float32)

    print('Predicting on {} cases'.format(len(ideologies)))

    #Build translators
    basic_translator = build_translator(seq2seq_basic_model_path, 
                                           config_path, baseline=True, min_length=min_length, max_length=max_length, 
                                            n_best=1, beam_size=beam_size)
    #app_translator   = build_translator(seq2seq_models_path + 'app_claim_context-d_embedding_step_120000.pt',
    #                                           config_path, baseline=False, min_length=min_length, max_length=max_length,
    #                                           n_best=1, beam_size=beam_size)

    app_bi_translator = build_translator(seq2seq_app_model_path,
                                           config_path,  min_length=min_length, max_length=max_length,
                                           n_best=1, beam_size=beam_size)


    #Perform translation
    basic_scores, basic_preds = basic_translator.translate(topics, 
                                                       src_dir=None, 
                                                       batch_size=16, 
                                                       attn_debug=False)


    # app_scores, app_preds    = app_translator.translate(topics, 
    #                                                  context_feats=users_embedding, 
    #                                                  key_phrase_feats=topic_key_phrases_embeddings,
    #                                                  src_dir=None, batch_size=16, attn_debug=False)

    if simple_context:
        app_bi_scores, app_bi_preds = app_bi_translator.translate(topics, 
                                                     context_feats=users_big_issues, 
                                                     src_dir=None, batch_size=16, attn_debug=False)
    else:

        glove_encoder = utils.load_glove_model('/workspace/ceph_data/glove/glove.42B.300d.txt')
        
        #if topic_key_phrases_path:
        topic_info_df = pd.read_parquet(topic_key_phrases_path)
        #topic_key_phrases_sents_dict = pd.Series(topic_info_df.keyphrase_sentence.values, index=topic_info_df.topic).to_dict()
        #topic_key_phrases_dict       = dict([(x[0], [y[0] for y in x[1]]) for x in topic_key_phrases_sents_dict.items()])
        #topic_key_sents_dict         = dict([(x[0], get_keyphrases_sentences(x[1], topic_key_phrases_dict[x[0]], 2)) for x in topic_key_phrases_sents_dict.items()])
        topic_key_phrases_dict = pd.Series(topic_info_df.keyphrases.values, index=topic_info_df.topic).to_dict()
        dev_df['topic_keyphrases']      = dev_df.topic.apply(lambda x: topic_key_phrases_dict[x])
        #dev_df['topic_keysentences']    = dev_df.topic.apply(lambda x: topic_key_sents_dict[x])
        dev_df['key_phrases_embeddings']= dev_df.topic_keyphrases.apply(lambda keyphrases: [utils.encode_phrase(x, glove_encoder) for x in keyphrases[0:5] if isinstance(x, str)])
        topic_key_phrases_embeddings = np.array(dev_df['key_phrases_embeddings'].tolist())
        #dev_df['key_phrases_embeddings']   = dev_df.key_phrases_embeddings.apply(lambda x: [y for y in x if y is not None])
        #filter only topic that has at least key phrases
        #dev_df  = dev_df[dev_df.topic_keyphrases.map(lambda d: len(d)) > 0]

        app_bi_scores, app_bi_preds = app_bi_translator.translate(topics, 
                                                     context_feats=users_big_issues, 
                                                     key_phrase_feats=topic_key_phrases_embeddings,
                                                     src_dir=None, batch_size=16, attn_debug=False)


    baseline_trans = [x[0] for x in basic_preds]
    #app_trans = [x[0] for x in app_preds]
    app_bi_trans = [x[0] for x in app_bi_preds]


    dev_df['baseline_preds']   = baseline_trans
    #dev_df['embedding_preds']  = app_trans
    dev_df['big_issues_preds'] = app_bi_trans


    dev_df.to_csv(output_path)



if __name__ == "__main__":

   
    parser = argparse.ArgumentParser(description='seq2seq claim generation')
    parser.add_argument('--seq2seq_basic_model_path')
    parser.add_argument('--seq2seq_app_model_path')
    parser.add_argument('--training_dataset_path')
    parser.add_argument('--valid_dataset_path')
    parser.add_argument('--users_df_path')
    parser.add_argument('--output_path')
    parser.add_argument('--topic_key_phrases_path', default=None)
    parser.add_argument('--min_length', type=int, default=35)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=10)

    args = parser.parse_args()


    gen_preds(args.valid_dataset_path, args.training_dataset_path, 
         args.users_df_path, args.seq2seq_basic_model_path, args.seq2seq_app_model_path, args.output_path, 
         topic_key_phrases_path=args.topic_key_phrases_path,
         min_length=args.min_length,
         max_length=args.max_length,
         beam_size=args.beam_size)