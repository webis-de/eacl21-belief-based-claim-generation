import itertools
import json
import random
import re
import argparse

import numpy as np
import pandas as pd
import requests
import spacy
from nltk import RegexpParser, ngrams, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


class ArgsMeArgumentRetriever():

    # EXAMPLE QUERY: https://www.args.me/api/v2/arguments?query=gender+pay+gap
    API_URL = 'https://www.args.me/api/v2/arguments?query='
    HEADERS = {'Content-Type': 'application/json'}
    STOP_WORDS = set(stopwords.words('english'))
    NLP = spacy.load("en_core_web_sm", disable=[
                     "tagger", "parser", "ner", "textcat", "tokenizer"])
    NLP.add_pipe(NLP.create_pipe('sentencizer'))

    def __init__(self, keyphrases: bool, aspect_occurances: bool, synset: bool, topic_signatures: bool, 
        argsme_corpus_path: str, sample_size: int, stance: str=None):
        self.keyphrases = keyphrases
        self.aspect_occurances = aspect_occurances
        self.synset = synset
        self.topic_signatures = topic_signatures
        self.argsme_corpus_path = argsme_corpus_path
        self.sample_size = sample_size
        self.extracted_signatures = [[],[]]
        self.stance = stance

        if self.topic_signatures:
            unrelevant_arguments = self.sample_unrelevant_arguments()
            self.unrelevant_argument_ngrams = list(itertools.chain(*[self.extract_ngrams(text, 0) for text in unrelevant_arguments]))

    def find_aspect_occurance_sentences(self, argument_dict: dict) -> list:
        topic_aspect_sentences = []
        for sentence in self.NLP(argument_dict['premise']).sents:
            for aspect in argument_dict['aspects']+ self.extracted_signatures[0]:
                if aspect.lower() in sentence.text.lower():
                    if sentence.text not in topic_aspect_sentences:
                        topic_aspect_sentences.append((aspect, sentence.text))
        return topic_aspect_sentences

    def find_aspect_occurance_keyphrases(self, argument_dict: dict) -> list:
        topic_keyphrases = []
        for sentence in self.extract_keyphrases(argument_dict['premise']):
            for aspect in argument_dict['aspects']+ self.extracted_signatures[0]:
                if aspect.lower() in sentence.lower():
                    if sentence not in topic_keyphrases:
                        if self.keyphrases:
                            topic_keyphrases.append((sentence, aspect))
        return topic_keyphrases

    def extract_keyphrases(self, sentence: str) -> list:
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
                             tag in subtree.leaves() if not word in self.STOP_WORDS]
                if 1 <= len(keyphrase) <= 10:
                    keyphrases.append(' '.join(keyphrase).lower())

        return keyphrases

    def re_sample_non_relevant_args(slef):
        unrelevant_arguments = self.sample_unrelevant_arguments()
        self.unrelevant_argument_ngrams = list(itertools.chain(*[self.extract_ngrams(text, 0) for text in unrelevant_arguments]))

    def get_synsets(self, word: str) -> list:
        synonyms = []
        antonyms = []
        hypernyms = []
        hyponyms = []

        for syn in wordnet.synsets(word):
            for l in syn.hypernyms():
                for k in l.lemmas():
                    hypernyms.append(k.name())
            for l in syn.hyponyms():
                for k in l.lemmas():
                    hyponyms.append(k.name())
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

        return [*list(set(synonyms)), *list(set(antonyms)), *list(set(hypernyms)), *list(set(hyponyms))]

    def sample_unrelevant_arguments(self) -> list:
        with open(self.argsme_corpus_path, 'r') as fp:
            argsme_dict = json.load(fp)['arguments']
        keys = random.sample(list(argsme_dict), self.sample_size)
        return [argument['premises'][0]['text'] for argument in keys]

    def extract_ngrams(self, text: str, n: int) -> list:
        ngrams = []
        try:
            tokens = word_tokenize(
                re.sub(r'[^A-Za-z0-9]+', ' ', text.lower()))
            #remove tokens that are not alpha-numeric and less than 4 characters.
            tokens = [x for x in tokens if len(x) > 3 and x.isalpha()]
            ngrams = [tokens[i:i+n+1] for i in range(len(tokens)-(n))]
            #print(ngrams)
            return [' '.join(grams) for grams in ngrams]
        except:
            pass

    def extract_topic_signatures(self, query_arguments_dict: dict, max_ngram: int, num_args_retrieved: int, threshold: int) -> list:
        premises = []
        for argument in query_arguments_dict.items():
            if self.stance != None and argument[1]['stance'] != self.stance:
                continue
            topic_premise = argument[1]['premise']
            premises.append(topic_premise)
        topic_signatures = []
        synsets = []
        for n in range(max_ngram):
            relevant_argument_ngrams = list(itertools.chain(*[self.extract_ngrams(text, 0) for text in premises]))

            for relevant_argument_ngram in list(set(relevant_argument_ngrams)):
                o11 = relevant_argument_ngrams.count(relevant_argument_ngram)
                o21 = len(relevant_argument_ngrams)-o11
                o12 = self.unrelevant_argument_ngrams.count(relevant_argument_ngram)
                o22 = len(self.unrelevant_argument_ngrams)-o12
                N = sum([o11, o12, o21, o22])
                likelihood_ratio = 2*N*(
                    (o11/N)*np.log2((o11*N+1)/((o11+o12)*(o11+o21)+1))
                    + (o21/N)*np.log2((o21*N+1)/((o21+o22)*(o11+o21)+1))
                    + (o12/N)*np.log2((o12*N+1)/((o11+o12)*(o12+o22)+1))
                    + (o22/N)*np.log2((o22*N+1)/((o21+o22)*(o12+o22)+1))
                )
                if likelihood_ratio > threshold/(n+1):
                    if n == 0:
                        synsets += self.get_synsets(relevant_argument_ngram)
                    topic_signatures.append(
                        ' '.join([w for w in relevant_argument_ngram.split() if not w in self.STOP_WORDS]))
        topic_signatures = list(set(topic_signatures))
        if self.synset:
            synsets = list(set(synsets))
        return [topic_signatures, synsets]


    def compute_topic_sign(self, relevant_docs, max_ngram=1, threshold=10):
        topic_signatures = {}
        synsets = []
        for n in range(max_ngram):
            relevant_argument_ngrams = list(itertools.chain(*[self.extract_ngrams(text, 0) for text in relevant_docs]))

            for relevant_argument_ngram in list(set(relevant_argument_ngrams)):
                o11 = relevant_argument_ngrams.count(relevant_argument_ngram)
                o21 = len(relevant_argument_ngrams)-o11
                o12 = self.unrelevant_argument_ngrams.count(relevant_argument_ngram)
                o22 = len(self.unrelevant_argument_ngrams)-o12
                N = sum([o11, o12, o21, o22])
                likelihood_ratio = 2*N*(
                    (o11/N)*np.log2((o11*N+1)/((o11+o12)*(o11+o21)+1))
                    + (o21/N)*np.log2((o21*N+1)/((o21+o22)*(o11+o21)+1))
                    + (o12/N)*np.log2((o12*N+1)/((o11+o12)*(o12+o22)+1))
                    + (o22/N)*np.log2((o22*N+1)/((o21+o22)*(o12+o22)+1))
                )
                if likelihood_ratio > threshold/(n+1):
                    if n == 0:
                        synsets += self.get_synsets(relevant_argument_ngram)
                    
                    topic_sign_word = ' '.join([w for w in relevant_argument_ngram.split() if not w in self.STOP_WORDS])
                    topic_signatures[topic_sign_word] =(likelihood_ratio, o11/len(relevant_argument_ngrams))

        #topic_signatures = list(set(topic_signatures))
        if self.synset:
            synsets = list(set(synsets))
        
        return topic_signatures, synsets


    def extract_topic_pro_con_signatures(self, topic: str, max_ngram: int, num_args_retrieved: int, threshold: int, num_of_words: int = 10) -> list:
        query_arguments_dict = self.get_relevant_arguments(topic, num_args_retrieved)

        pro_premises = []
        con_premises = []
        all_premises = []
        for argument in query_arguments_dict.items():
            if argument[1]['stance'] == 'PRO':
                topic_premise = argument[1]['premise']
                pro_premises.append(topic_premise)

            if argument[1]['stance'] == 'CON':
                topic_premise = argument[1]['premise']
                con_premises.append(topic_premise)
            
            all_premises.append(argument[1]['premise'])

        print('Number of pro arguments:', len(pro_premises))
        print('Number of con arguments:', len(con_premises))
        topic_pro_signatures, _ = self.compute_topic_sign(pro_premises, max_ngram, threshold)
        topic_con_signatures, _ = self.compute_topic_sign(con_premises, max_ngram, threshold)
        #topic_signatures, _     = self.compute_topic_sign(all_premises, max_ngram, threshold)
        
        #compute uniquness across pro/con sides
        topic_pro_signatures = {x[0]: (x[1][0], x[1][1]/topic_con_signatures[x[0]][1]) if x[0] in topic_con_signatures else (x[1][0], 10000.0) for x in topic_pro_signatures.items()}
        topic_con_signatures = {x[0]: (x[1][0], x[1][1]/topic_pro_signatures[x[0]][1]) if x[0] in topic_pro_signatures else (x[1][0], 10000.0) for x in topic_con_signatures.items()}

        p = [x for x in topic_pro_signatures.items() if x[1][1] > 1.0]
        c = [x for x in topic_con_signatures.items() if x[1][1] > 1.0]

        p = [x[0] for x in sorted(p, key=lambda x: -x[1][0])[0:num_of_words]]
        c = [x[0] for x in sorted(c, key=lambda x: -x[1][0])[0:num_of_words]]
        
        return p, c, topic_pro_signatures, topic_con_signatures

    def get_relevant_arguments(self, topic, num_args_retrieved):
        query_arguments_dict = {}
        # extract content words
        topic_tokens = word_tokenize(re.sub(r'[^A-Za-z0-9]+', ' ', topic.lower()))
        topic_tokens = [w for w in topic_tokens if not w in self.STOP_WORDS]

        print('CALLING: ', self.API_URL+'+'.join(topic_tokens)+'&pageSize='+str(num_args_retrieved))
        # get response
        response = requests.get(
            self.API_URL+'+'.join(topic_tokens)+'&pageSize='+str(num_args_retrieved), headers=self.HEADERS)

        # process response
        if response.status_code == 200:
            response_content = json.loads(response.content.decode('utf-8'))
            print('Number of retrieved arguments: ', len(response_content['arguments']))
            for i, argument in enumerate(response_content['arguments']):
                argument_dict = {
                    'conclusion': argument['conclusion'],
                    'premise': argument['premises'][0]['text'],
                    'aspects': [argument['context']['aspects'][i]['name'] for i in range(len(argument['context']['aspects']))],
                    'score':  argument['explanation']['score'],
                    'stance': argument['stance']
                }
                if self.aspect_occurances:
                    argument_dict['argument_aspect_sentences'] = self.find_aspect_occurance_sentences(argument_dict)
                    argument_dict['argument_aspect_keyphrases'] = self.find_aspect_occurance_keyphrases(argument_dict)
                if argument['id'] not in query_arguments_dict:
                    query_arguments_dict[argument['id']] = argument_dict
        elif response.status_code == 400:
            print('Bad Request!')
        elif response.status_code == 500:
            print('Server Error!')
        else:
            print('Unknown Error!')

        return query_arguments_dict


    def retrieve_arguments(self, query: str, num_args_retrieved: int, threshold: int) -> dict:
        query_arguments_dict = {}

        # treat each sentence of the query as an independent query
        for sentence in self.NLP(query).sents:
            # extract content words
            word_tokens = word_tokenize(
                re.sub(r'[^A-Za-z0-9]+', ' ', sentence.text.lower()))
            filtered_sentence = [
                w for w in word_tokens if not w in self.STOP_WORDS]

            print('CALLING: ', self.API_URL+'+'.join(filtered_sentence)+'&pageSize='+str(num_args_retrieved))
            # get response
            response = requests.get(
                self.API_URL+'+'.join(filtered_sentence)+'&pageSize='+str(num_args_retrieved), headers=self.HEADERS)

            # process response
            if response.status_code == 200:
                response_content = json.loads(response.content.decode('utf-8'))
                print('Number of retrieved arguments: ', len(response_content['arguments']))
                for i, argument in enumerate(response_content['arguments']):
                    argument_dict = {
                        'conclusion': argument['conclusion'],
                        'premise': argument['premises'][0]['text'],
                        'aspects': [argument['context']['aspects'][i]['name'] for i in range(len(argument['context']['aspects']))],
                        'score':  argument['explanation']['score'],
                        'stance': argument['stance']
                    }
                    if self.aspect_occurances:
                        argument_dict['argument_aspect_sentences'] = self.find_aspect_occurance_sentences(argument_dict)
                        argument_dict['argument_aspect_keyphrases'] = self.find_aspect_occurance_keyphrases(argument_dict)
                    if argument['id'] not in query_arguments_dict:
                        query_arguments_dict[argument['id']] = argument_dict
            elif response.status_code == 400:
                print('Bad Request!')
            elif response.status_code == 500:
                print('Server Error!')
            else:
                print('Unknown Error!')

            if self.topic_signatures:
                self.topic_signatures = False
                self.extracted_signatures = self.extract_topic_signatures(query_arguments_dict, 1, num_args_retrieved, threshold)
                print('Topic Signiture: ',  self.extracted_signatures[0])
                if self.synset:
                    print(self.extracted_signatures[1])
                    query_arguments_dict.update(self.retrieve_arguments(' '.join(self.extracted_signatures[0])+' '+' '.join(self.extracted_signatures[1]), threshold=threshold, num_args_retrieved=40))
                else:
                    query_arguments_dict.update(self.retrieve_arguments(' '.join(self.extracted_signatures[0]), threshold=threshold, num_args_retrieved=num_args_retrieved))
                query_arguments_dict['topic_signature_words'] = self.extracted_signatures
        #self.extracted_signatures = [[],[]]

        return query_arguments_dict


    def reset_model(self):
        self.topic_signatures =True
        self.extracted_signatures = [[],[]]

def get_topic_sents(topic_key, topic_arguments_dict):
    topic = topic_arguments_dict[topic_key]

    sentences = []
    for key, arg in topic.items():
        if key != 'topic_signature_words':
            sentences.append(arg['argument_aspect_sentences'])
    sentences = list(itertools.chain(*sentences))
    sentences = [x[1] for x in sentences]
    sentences = list(set(sentences))
    
    return sentences

def get_topic_keyphrase_sentences(topic_key, topic_arguments_dict):
    topic = topic_arguments_dict[topic_key]

    sentences = []
    for key, arg in topic.items():
        if key != 'topic_signature_words':
            sentences.append(arg['argument_aspect_sentences'])
    sentences = list(itertools.chain(*sentences))
    
    key_phrase_sents_dict = {}
    for item in sentences:
        if item[0].lower() not in key_phrase_sents_dict:
            key_phrase_sents_dict[item[0].lower()] = [item[1]]
        else:
            key_phrase_sents_dict[item[0].lower()].append(item[1])

    key_phrase_sents = [[x[0], ' <:SENT:> '.join(set(x[1]))] for x in key_phrase_sents_dict.items()]
    return key_phrase_sents

def get_topic_keyphrases(topic_key, topic_arguments_dict):
    topic = topic_arguments_dict[topic_key]

    keyphrases = []
    for key, arg in topic.items():
        if key != 'topic_signature_words':
            keyphrases.append(arg['argument_aspect_keyphrases'])
    keyphrases = list(itertools.chain(*keyphrases))
    keyphrases = [x[0] for x in keyphrases]
    keyphrases = list(set(keyphrases))
    
    return keyphrases

def get_topic_signature(topic_key, topic_arguments_dict):
    topic = topic_arguments_dict[topic_key]
    return topic['topic_signature_words'][0]


def extend_topics_info(topics_df_path, topic_arguments_dict_path, output_path=None):
    
    topics_df = pd.read_pickle(topics_df_path)
    topic_arguments_dict = json.load(open(topic_arguments_dict_path, encoding='latin1'))


    topics_df['sentences'] = topics_df.topic.apply(lambda x: get_topic_sents(x, topic_arguments_dict))
    topics_df['keyphrases'] = topics_df.topic.apply(lambda x: get_topic_keyphrases(x, topic_arguments_dict))    
    topics_df['keyphrase_sentence'] = topics_df.topic.apply(lambda x: get_topic_keyphrase_sentences(x, topic_arguments_dict))
    topics_df['topic_signatures'] = topics_df.topic.apply(lambda x: get_topic_signature(x, topic_arguments_dict))
    
    if output_path == None:
        topics_df.to_parquet(topics_df_path+'.extended.gzip', compression='gzip')
    else:
        topics_df.to_parquet(output_path, compression='gzip')

def build_topic_dictionary(topics_path, argsme_corpus, output_path, num_args_retrieved=20, threshold=300, stance=None, resample_non_relevant_args=False):

    df = pd.read_pickle(topics_path)

    amar = ArgsMeArgumentRetriever(keyphrases=True,
                            aspect_occurances=True,
                            synset=False,
                            topic_signatures=True,
                            argsme_corpus_path=argsme_corpus,
                            sample_size=1000,
                            stance=stance)

    topic_args = {}
    for topic in tqdm(df['topic']):
        args = amar.retrieve_arguments(query=topic, num_args_retrieved=num_args_retrieved, threshold=threshold)
        topic_args[topic]= args
        amar.reset_model()
        if resample_non_relevant_args:
            amar.re_sample_non_relevant_args()

    with open(output_path, 'w') as fp:
        json.dump(topic_args, fp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='argument retrieval')
    parser.add_argument('task')
    parser.add_argument('--topic_path')
    parser.add_argument('--argsme_corpus_path')
    parser.add_argument('--num_args_retrieved', type=int, default=20)
    parser.add_argument('--threshold', type=int, default=300)
    parser.add_argument('--stance', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--resample_non_relevant_args', action='store_true')
    parser.add_argument('--topic_arguments_dict_path')

    args = parser.parse_args()

    if args.task == 'argument_retrieval':
        build_topic_dictionary(args.topic_path, args.argsme_corpus_path, args.topic_arguments_dict_path, 
            args.num_args_retrieved, args.threshold, args.stance, args.resample_non_relevant_args)
    
    if args.task == 'extend_topics_df':
        extend_topics_info(args.topic_path, args.topic_arguments_dict_path, args.output_path)



    ## GET KEYPHRASES FOR SINGLE TOPIC
    ###################################################################################################################
    # amar = ArgsMeArgumentRetriever(keyphrases=True,
    #                         aspect_occurances=True,
    #                         synset=False,
    #                         topic_signatures=True,
    #                         argsme_corpus_path='/workspace/ceph_data/data/argsme_data/args-me.json',
    #                         sample_size=1000)
    # args = amar.retrieve_arguments(query="Should there be a gender pay gap?", num_args_retrieved=20, threshold=300)
    # print(len(args))

    # sentences = []
    # for key, arg in args.items():
    #     if key != 'topic_signature_words':
    #         sentences += arg['argument_aspect_keyphrases']
    # sentences = list(itertools.chain(*sentences))

    # print(list(set(sentences)))
    # print(args.keys())

    # CREATE TOPIC DICTIONARY THAT CONTAINS ALL THE INFORMATION RETRIEVED
    ##################################################################################################################
    

    # GET SENTENCES AND KEYPHRASES FROM THE TOPIC DICTIONARY
    ##################################################################################################################
    # with open('/workspace/ceph_data/data/argument_retrieval_data/topic_arguments_dict.json', 'r') as fp:
    #     topic_arguments_dict = json.load(fp)
    # args = topic_arguments_dict
    
    # df_sentences = []
    # for key, topic in args.items():
    #     sentences = []
    #     for key, arg in topic.items():
    #         if key != 'topic_signature_words':
    #             sentences.append(arg['argument_aspect_sentences'])
    #     sentences = list(itertools.chain(*sentences))
    #     sentences = list(set(sentences))
    #     df_sentences.append(sentences)

    # df_keyphrases = []
    # for key, topic in args.items():
    #     keyphrases = []
    #     for key, arg in topic.items():
    #         if key != 'topic_signature_words':
    #             keyphrases += arg['argument_aspect_keyphrases']
    #     keyphrases = list(itertools.chain(*keyphrases))
    #     keyphrases = list(set(keyphrases))
    #     df_keyphrases.append(keyphrases)

    # df = pd.read_csv('/workspace/ceph_data/data/preprocessed_data/topics.csv')
    # df['argument_aspect_sentences'] = df_sentences
    # df['argument_aspect_keyphrases'] = df_keyphrases
    # df.to_parquet('/root/Desktop/workspace/ceph_data/data/argument_retrieval_data/topics.parquet.gzip', compression='gzip')
    # print(df.head(1))


