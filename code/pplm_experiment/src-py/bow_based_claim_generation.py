import sys
sys.path.insert(0, "../../src-py/")

import os
import utils
import pickle
from run_pplm_discrim_train import *
from run_pplm import *

from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

import spacy

nlp = spacy.load("en_core_web_sm")


device = 'cuda:0'

big_issues = [
 'Abortion',
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
 'Welfare'
]

used_big_issues = [
    'Abortion',
    'Death Penalty',
    'Gay Marriage',
    'Drug Legalization',
    'Gun Rights',
    'Global Warming Exists',
    'Environmental Protection',
    'Smoking Ban',
    'Minimum Wage',
    'Border Fence',
    'Social Programs',
    'Internet Censorship',
    'Capitalism',
    'Flat Tax',
    'Progressive Tax',
    'Socialism',
    'Free Trade',
    'Globalization'
]

bi_mask = [big_issues.index(x) for x in used_big_issues]

def get_model(name):
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name, output_hidden_states=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.to(device)
    model.eval()
    return model, tokenizer

    
def generate_belief_based_text_via_discrim(topic, belief, model, tokenizer, discrim_model, 
                               length=25, stepsize=0.01, num_iterations=3, perturb=True):
    tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + topic)
    
    set_generic_model_params(discrim_model + '/belief_discrim_classifier_head_epoch_10.pt', 
                             discrim_model + '/belief_discrim_classifier_head_meta.json')
    
    classifier, class_id = get_classifier('generic', belief, device)

    pert_gen_tok_text, _, _ = generate_text_pplm(model=model, tokenizer=tokenizer,
            context=tokenized_cond_text, device=device, perturb=perturb,
            bow_indices=[], classifier=classifier, class_label=class_id, loss_type=2,
            length=length, stepsize=stepsize, num_iterations=num_iterations, grad_length=30)
    
    return tokenizer.decode(pert_gen_tok_text.tolist()[0])


def generate_belief_based_text_via_bow(topic, belief_bows, model, tokenizer, perturb=True,
                                      length=25, stepsize=0.01, num_iteration=3,
                                      kl_scale=0.01, gamma=1.5, grad_length=25, 
                                      decay=False, window_length=0, repetition_penalty=1.0):
    

    print('Generate preds for: ', topic)

    bow_indices = []
    for belief_bow in belief_bows:
        bow_indices.append([tokenizer.encode(word.strip(), add_prefix_space=True) for word in belief_bow])
    
    tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + topic)
    pert_gen_tok_text, _, _ = generate_text_pplm(model=model, tokenizer=tokenizer, context=tokenized_cond_text, 
                                                 device=device, bow_indices=bow_indices, loss_type=1,
                                                 length=length, stepsize=stepsize, num_iterations=num_iteration,
                                                 perturb=perturb,
                                                 kl_scale=kl_scale,
                                                 gamma=gamma,
                                                 grad_length=grad_length,
                                                 decay=decay,
                                                 window_length=window_length,
                                                 repetition_penalty=repetition_penalty)

    return tokenizer.decode(pert_gen_tok_text.tolist()[0])


def user_bow_from_big_issues_stances_dict(user_bi_dict, pro_bows, con_bows, apply_mask=False):
    user_bow = []
    for bi in user_bi_dict.items():
        if bi[1] == 1 and bi[0] in con_bows:
            user_bow += con_bows[bi[0]]
        if bi[1] == 3 and bi[0] in pro_bows:
            user_bow += pro_bows[bi[0]]
    

    #print(user_bi_dict)
    #print(pro_bows)
    #print(con_bows)
    #print(user_bow)

    return list(set(user_bow))

def user_bow_from_big_issues_stances(user_bi, pro_bows, con_bows, apply_mask=False):
    user_bow = []
    for i, bi in enumerate(user_bi):
        if apply_mask and i not in bi_mask:
            continue
        if bi in [1,3]:
            bi_bow = list(pro_bows[i]) if bi == 3 else list(con_bows[i])
            user_bow += bi_bow

    return list(set(user_bow))

def get_modified_topic(topic):
        doc = nlp(topic)
        
        nps = [str(x) for x in doc.noun_chunks]
        
        if len(nps) == 0:
            return topic

        return sorted(nps, key=lambda x: - len(x))[0]
        


def gen_belief_based_claims(pretrained_model_path, training_dataset_path, valid_dataset_path, output_path,
    users_df_path,
    big_issues_bows_path=None,
    user_bows_path=None,
    length=15,
    stepsize=0.15,
    num_iteration=3,
    gamma=0.5,
    kl_scale=0.05,
    decay=False,
    window_length=5,
    sample=None,
    repetition_penalty=1.5,
    apply_mask=False,
    modify_topic=False):
    
    grad_length = length


    ft_model, ft_tokenizer = get_model(pretrained_model_path)



    #Load dataset
    train_df= pd.read_csv(training_dataset_path)
    valid_df  = pd.read_csv(valid_dataset_path)


    #Map user's ideologies to beliefs
    valid_df['user_belief'] = valid_df.user_ideology.apply(lambda x: utils.ideology_map[x])
    

    #Filter only users that are in the training dataset and doesn't not have unkown belief
    training_users = set(train_df.user.tolist())
    valid_df = valid_df[valid_df.user.isin(training_users)]
    #valid_df = valid_df[valid_df.user_belief != 'unkown']

    #Get user's big-issues vectors
    users_info = pd.read_pickle(users_df_path)
    users_bi   = pd.Series(users_info.big_issues.values, index=users_info.user).to_dict()

    valid_df['user_bi']      = valid_df.user.apply(lambda x: users_bi[x])    
    valid_df['user_bi_dict'] = valid_df.user_bi.apply(lambda x: utils.desc_user_from_vec(x))

    #user_bow from their arguments
    if user_bows_path != None:
        user_bows = pickle.load(open(user_bows_path, 'rb'))
        valid_df['user_bow']= valid_df.user.apply(lambda x: user_bows[x])

    #user_bow from their big_issues vector
    if big_issues_bows_path != None:
        bi_bows_df = pd.read_pickle(big_issues_bows_path)
        #pro_bows = bi_bows_df.pro_bow.tolist()
        #con_bows = bi_bows_df.con_bow.tolist()
        pro_bows_dict = pd.Series(bi_bows_df.pro_bow.values, index=bi_bows_df.topic).to_dict()
        con_bows_dict = pd.Series(bi_bows_df.con_bow.values, index=bi_bows_df.topic).to_dict()

        valid_df['user_bi_bow'] = valid_df.apply(lambda row: 
            user_bow_from_big_issues_stances_dict(row['user_bi_dict'], pro_bows_dict, con_bows_dict, apply_mask=apply_mask), axis=1)


    if sample != None:
        valid_df = valid_df.sample(sample)


    valid_df['modified_topic'] = valid_df.topic.apply(lambda x: get_modified_topic(x))

    print('Predicting on {} cases'.format(len(valid_df)))
    
    valid_df['uncond_pred_claim'] = valid_df.apply(lambda row: 
                                                 generate_belief_based_text_via_bow(row['modified_topic'] if modify_topic else row['topic'], [], 
                                                                                    ft_model, ft_tokenizer, 
                                                                                    length=length, 
                                                                                    stepsize=0.0,
                                                                                    perturb=False,
                                                                                    num_iteration=num_iteration
                                                ) , axis=1)


    print('Generate preds based on users bows......')
    if user_bows_path:
        valid_df['user_bow_based_pred_claim'] = valid_df.apply(lambda row: 
                                                 generate_belief_based_text_via_bow(row['modified_topic'] if modify_topic else row['topic'], [row['user_bow']], 
                                                                                    ft_model, ft_tokenizer, 
                                                                                    length=length, 
                                                                                    stepsize=stepsize, 
                                                                                    num_iteration=num_iteration,
                                                                                    gamma=gamma, grad_length=grad_length,
                                                                                    kl_scale=kl_scale, decay=decay, 
                                                                                    window_length=window_length, 
                                                                                    repetition_penalty=repetition_penalty
                                                ) , axis=1)


    print('Generate preds based on users bi bows......')
    if big_issues_bows_path:
        valid_df['user_bi_bow_based_pred_claim'] = valid_df.apply(lambda row: 
                                                 generate_belief_based_text_via_bow(row['modified_topic'] if modify_topic else row['topic'], [row['user_bi_bow']], 
                                                                                    ft_model, ft_tokenizer, 
                                                                                    length=length, 
                                                                                    stepsize=stepsize, 
                                                                                    num_iteration=num_iteration,
                                                                                    gamma=gamma, grad_length=grad_length,
                                                                                    kl_scale=kl_scale, decay=decay, 
                                                                                    window_length=window_length, 
                                                                                    repetition_penalty=repetition_penalty
                                                ) , axis=1)


    valid_df.to_csv(output_path)



if __name__ == "__main__":

   
    parser = argparse.ArgumentParser(description='pplm claim generation')
    parser.add_argument('--pretrained_model_path')
    parser.add_argument('--training_dataset_path')
    parser.add_argument('--valid_dataset_path')
    parser.add_argument('--users_df_path')
    parser.add_argument('--output_path')
    parser.add_argument('--big_issues_bows_path', default=None)
    parser.add_argument('--user_bows_path', default=None)
    parser.add_argument('--length', type=int, default=15)
    parser.add_argument('--sample_data', type=int, default=None)
    parser.add_argument('--stepsize', type=float, default=0.15)
    parser.add_argument('--window_length', type=float, default=0)
    parser.add_argument('--num_iteration', type=int, default=3)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--apply_mask', action='store_true', default=False)
    parser.add_argument('--modify_topic', action='store_true', default=False)

    args = parser.parse_args()


    gen_belief_based_claims(args.pretrained_model_path, 
                            args.training_dataset_path, 
                            args.valid_dataset_path, 
                            args.output_path,
                            args.users_df_path,
                            big_issues_bows_path=args.big_issues_bows_path,
                            user_bows_path=args.user_bows_path,
                            length=args.length,
                            stepsize=args.stepsize,
                            num_iteration=args.num_iteration,
                            sample=args.sample_data,
                            repetition_penalty=args.repetition_penalty,
                            apply_mask=args.apply_mask,
                            window_length=args.window_length,
                            modify_topic=args.modify_topic)