import sys
import json

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

from proc_data import coordinate, build_prompt_input
from prompt_emb_layer import PromptEmbedding, PromptDecoder

from eval_task_adp import (
    load_train_data, load_adv_eval, load_base_eval
)

def load_eval_data(dataset_name, eval_split):
    _, _, _, dformat = load_train_data(dataset_name)

    if eval_split == 'adv':
        sent1_list, sent2_list, label_list = load_adv_eval(dataset_name, dformat)
    
    elif eval_split == 'base':
        sent1_list, sent2_list, label_list, _ = load_base_eval(
            dataset_name, split='dev', dformat=dformat
        )
    
    return sent1_list, sent2_list, label_list, dformat


def get_base_logits(tok, model, t_idx, f_idx, ok_idx,
                    num_prompt, prompt_str = None, mlm = True):
    if not mlm:
        return 0, 0, 0
    
    if prompt_str is None:
        input_txt = [
            'It is [MASK] that true.',
            'It is [MASK] that frue.',
            'It is [MASK] that ok.'
        ]
        offset = 0
    else:
        input_txt = [
            f'{prompt_str} It is [MASK] that .',
            f'{prompt_str} It is [MASK] that .',
            f'{prompt_str} It is [MASK] that .'
        ]
        offset = num_prompt
    input_enc = tok(
        input_txt, return_tensors = 'pt',
        padding = 'longest', truncation = True
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(input_ids, attn_mask)

    t_base = result.logits[0][offset + 3][t_idx].item()
    f_base = result.logits[1][offset + 3][f_idx].item()
    ok_base = result.logits[2][offset + 3][ok_idx].item()
    
    return t_base, f_base, ok_base


def mlm_evaluate(tok, model, input_list, rvs_flag,
             t_idx = None, f_idx = None, ok_idx = None,
             t_base = None, f_base = None, ok_base = None,
             model_type = 'mlm', num_prompt = None, mnli = False
    ):
    
    input_enc = tok(
        input_list,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(
            input_ids = input_ids,
            attention_mask = attn_mask
        )
    
    _, pred = result.logits.max(2)

    if num_prompt is None:
        offset = 0
    else:
        offset = num_prompt

    # print(input_ids[0][offset + 3])
    # abort()
    
    true_logits = result.logits[:, offset + 3, t_idx] - t_base
    false_logits = result.logits[:, offset + 3, f_idx] - f_base
    ok_logits = result.logits[:, offset + 3, ok_idx] - ok_base

    if rvs_flag == 0:
        return true_logits, false_logits #, ok_logits
    elif rvs_flag == 1:
        return false_logits, true_logits #, ok_logits
    else:
        print('Rvs_flag not supported')
        abort()


def cls_evaluate(
        tok, model, input_list, rvs_flag,
        t_idx = None, f_idx = None, ok_idx = None,
        t_base = None, f_base = None, ok_base = None,
        model_type='sc', num_prompt = None, mnli = False,
    ):
    
    input_enc = tok(
        input_list,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(
            input_ids = input_ids,
            attention_mask = attn_mask
        )
    
    logits = result.logits
    logits = F.softmax(result.logits, dim = -1)
    
    true_logits = logits[:, 0]
    false_logits = logits[:, -1]

    if mnli:
        ok_logits = logits[:, 1]
        return true_logits, ok_logits, false_logits

    # if model_type == 'sc':
    #     return true_logits, false_logits

    if rvs_flag == 0:
        return true_logits, false_logits
    else:
        return false_logits, true_logits


def gen_prompt_tok(num_prompt):
    prompt_tokens = [f'<prompt_token_{i}>' for i in range(num_prompt)]
    return prompt_tokens


def add_prompt_layer(model, dataset_name, num_prompt, model_type_str):
    if model_type_str == 'bert':
        model.bert.embeddings.word_embeddings = PromptEmbedding(
            model.bert.embeddings.word_embeddings, num_prompt,
            # f'model_ft_file/{dataset_name}_prompt_emb.pt'
            prompt_path = f'model_ft_file/mnli_prompt_emb_binary.pt'
        )

        model.cls.predictions.decoder = PromptDecoder(
            model.cls.predictions.decoder,
            num_prompt,
            model.bert.embeddings.word_embeddings.prompt_emb
        )
    elif model_type_str == 'deberta':
        pass
    else:
        print(f'Model {model_type_str} not supported')
        abort()
    return model


def eval_prompt_seq_cls(
        dataset_name, eval_mode, model_tag, data_split,
        data = None, mlm = False, model_type = 'sc'
    ):

    tok = AutoTokenizer.from_pretrained(
        f'model_file/deberta-{model_tag}-tok.pt'
        # f'model_file/{model_type_str}-large-tok.pt'
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f'model_ft_file/cls_{dataset_name}_{model_tag}_{data_split}.pt'
    )
    
    model.cuda()
    model.eval()

    model = nn.DataParallel(model)

    sent1_list, sent2_list, label_list, dformat = load_eval_data(
        dataset_name, eval_mode
    )

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    num_crr = 0

    batch_size = 32

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        sent2_batch = sent2_list[i: i + batch_size]
        label_batch = torch.Tensor(label_list[i: i + batch_size]).long()

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli':
            score_board = torch.zeros(cur_bs, 3)
        else:
            score_board = torch.zeros(cur_bs, 2)

        for j in range(num_prompt_type):
            false_scores, true_scores, ok_scores = cls_evaluate(
                tok, model, prompt_input_group[j], rvs_map[j], model_type
            )

            # '''
            score_board[:, 0] += false_scores.cpu() # - f_base
            if dataset_name == 'mnli':
                score_board[:, 2] += true_scores.cpu() # - t_base
                score_board[:, 1] += ok_scores.cpu()
            else:
                score_board[:, 1] += true_scores.cpu()
            # '''

            # print(score_board)
            # abort()
            _, pred = score_board.max(1)
        
        _, pred = score_board.max(1)
        num_crr += (pred == label_batch).float().sum()

        '''
        print('')
        print(pred)
        print(label_batch)
        abort()
        # '''
    
    acc = num_crr / num_case
    
    print('------------------------')
    print(f'Accuracy = {acc}')
    print('------------------------\n')


def const_distance(f_logits_1, t_logits_1, f_logits_2, t_logits_2):
    log_prob_1 = F.log_softmax(torch.cat(
        [f_logits_1.unsqueeze(1), t_logits_1.unsqueeze(1)], dim = 1
    ), dim = 1)
    log_prob_2 = F.log_softmax(torch.cat(
        [t_logits_2.unsqueeze(1), f_logits_2.unsqueeze(1)], dim = 1
    ), dim = 1)
    dist = log_prob_1 - log_prob_2
    entropy_1 = log_prob_1[:, 0] - log_prob_1[:, 1]
    entropy_2 = log_prob_2[:, 0] - log_prob_2[:, 1]
    entropy = (entropy_1 * entropy_1 + entropy_2 * entropy_2).mean()
    return (dist * dist).mean() * 1 + entropy * 0.05


def eval_prompt_const(
        dataset_name, eval_mode, model_tag, data_split,
        data = None, mlm = False, model_type = 'sc'
    ):

    tok = AutoTokenizer.from_pretrained(
        f'model_file/deberta-{model_tag}-tok.pt'
        # f'model_file/{model_type_str}-large-tok.pt'
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f'model_ft_file/cls_{dataset_name}_{model_tag}_{data_split}.pt'
    )
    
    model.cuda()
    model.train()

    model = nn.DataParallel(model)

    sent1_list, sent2_list, label_list, dformat = load_eval_data(
        dataset_name, eval_mode
    )

    if data is not None:
        sent1_list = data['sent1_list']
        sent2_list = data['sent2_list']

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    total_loss = 0

    batch_size = 32

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        sent2_batch = sent2_list[i: i + batch_size]
        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        false_scores_1, true_scores_1 = cls_evaluate(
            tok, model, prompt_input_group[0], rvs_map[0], model_type
        )

        false_scores_2, true_scores_2 = cls_evaluate(
            tok, model, prompt_input_group[1], rvs_map[1], model_type
        )

        loss = const_distance(
            false_scores_1, true_scores_1, false_scores_2, true_scores_2
        )

        total_loss += loss * cur_bs

    print(f'Total_loss = {total_loss.item()}\n')
    return total_loss / num_case


def prompt_seq_cls_relabel(
        dataset_name, sent1_list, sent2_list, model_type = 'sc',
        tok = None, model = None, model_path = None,
        model_type_str='deberta', model_size_str='large',
        mlm=False, batch_size=16, num_prompt_type=1
    ):

    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            # f'model_file/deberta-large-tok.pt'
            f'model_file/{model_type_str}-{model_size_str}-tok.pt'
        )

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
    
    model.cuda()
    model.eval()

    model = nn.DataParallel(model)

    num_case = len(sent1_list)
    pred_list = []
    score_board_list = []

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        if sent2_list is None:
            sent2_batch = sent1_batch
        else:
            sent2_batch = sent2_list[i: i + batch_size]

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli':
            score_board = torch.zeros(cur_bs, 3)
        else:
            score_board = torch.zeros(cur_bs, 2)

        for j in range(0, num_prompt_type):
            
            false_scores, true_scores = cls_evaluate(
                tok, model,
                prompt_input_group[j],
                int(j==1 and rvs_map[1] != rvs_map[0]),
                model_type
            )

            score_board[:, 0] += false_scores.cpu() # - f_base
            if dataset_name == 'mnli':
                score_board[:, 2] += true_scores.cpu() # - t_base
            else:
                score_board[:, 1] += true_scores.cpu()

            _, pred = score_board.max(1)
        
        _, pred = score_board.max(1)
        pred_list += pred.tolist()
        score_board_list.append(score_board)
    
    return pred_list, torch.cat(score_board_list, dim=0)


if __name__ == '__main__':

    dataset_name = sys.argv[1]
    model_mode = sys.argv[2]
    eval_split = sys.argv[3]
    model_type = 'sc'

    num_prompt_type = int(sys.argv[4])
    num_prompt = int(sys.argv[5])
    model_config = sys.argv[6]
    exp_id = sys.argv[7]

    if model_config == 'pretrain':
        model_path = 'model_ft_file/mnli_model_sc_5e-06_binary_pb.pt'
        # model_path = 'model_ft_file/mnli_model_mlm_1e-05_binary_p1.pt'
        # model_path = 'model_ft_file/mnli_model_sc_5e-06_single_p0.pt'
    if model_config == 'self_train':
        model_path = f'model_ft_file/cls_{dataset_name}_large_syn_data_relabel_{exp_id}.pt'

    model_type_str = 'bert'

    tok = AutoTokenizer.from_pretrained(
        'model_file/bert-large-tok.pt'
        # 'model_file/deberta-large-tok.pt'
        # f'model_file/{model_type_str}-large-tok.pt'
    )
    t_idx = tok.convert_tokens_to_ids('true')
    f_idx = tok.convert_tokens_to_ids('false')
    ok_idx = tok.convert_tokens_to_ids('ok')

    if model_type == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(
            # 'model_file/deberta-large-mlm.pt'
            # f'model_file/{model_type_str}-large-mlm.pt'
            # 'model_ft_file/qnli_model_mt.pt'
            model_path
        )
        eval_func = mlm_evaluate
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            # 'model_file/deberta-large-mlm.pt'
            # f'model_file/{model_type_str}-large-mlm.pt'
            # 'model_ft_file/mnli_model_sc_5e-06.pt'
            # 'model_ft_file/mnli_model_sc_5e-06_binary_p1.pt'
            # 'model_ft_file/cls_qnli_large_syn_data_relabel.pt'
            model_path
        )
        eval_func = cls_evaluate
    
    if model_mode == 'mt':
        prompt_str = None
        num_prompt = None

    elif model_mode == 'pt':
        model = AutoModelForMaskedLM.from_pretrained(
            f'model_file/{model_type_str}-large-mlm.pt'
        )

        prompt_tok_list = gen_prompt_tok(num_prompt)
        tok.add_tokens(prompt_tok_list)
        prompt_str = ' '.join(prompt_tok_list)
        # print(prompt_str)

        model = add_prompt_layer(
            model, dataset_name, num_prompt, model_type_str
        )
    
    else:
        print(f'Model mode {model_mode} not supported.')
        abort()
    
    model.cuda()
    model.eval()

    sent1_list, sent2_list, label_list, dformat = load_eval_data(
        dataset_name, eval_split
    )

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    num_crr_0 = 0
    num_crr_1 = 0
    num_crr = 0

    batch_size = 32

    t_base, f_base, ok_base = get_base_logits(
        tok, model, t_idx, f_idx, ok_idx,
        num_prompt, prompt_str = prompt_str,
        mlm = (model_type == 'mlm')
    )

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        sent2_batch = sent2_list[i: i + batch_size]
        label_batch = torch.Tensor(label_list[i: i + batch_size]).long()

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch,
            mlm = (model_type == 'mlm')
        )
        # print(prompt_input_list[0])
        # abort()

        if model_mode == 'pt':
            prompt_input_list  = [
                f'{prompt_str} {x}' for x in prompt_input_list
            ]

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli':
            score_board = torch.zeros(cur_bs, 3).cuda()
        else:
            score_board_0 = torch.zeros(cur_bs, 2).cuda()
            score_board_1 = torch.zeros(cur_bs, 2).cuda()
            score_board = torch.zeros(cur_bs, 2).cuda()

        for j in range(0, num_prompt_type):
            '''false_scores, true_scores, ok_scores = mlm_evaluate(
                tok, model, prompt_input_group[j], rvs_map[j],
                t_idx, f_idx, ok_idx,
                t_base, f_base, ok_base,
                num_prompt = num_prompt
            )'''

            if dataset_name == 'mnli':
                false_scores, true_scores = eval_func(
                    tok, model, prompt_input_group[j], rvs_map[j],
                    t_idx = t_idx, f_idx = f_idx, ok_idx = ok_idx,
                    t_base = t_base, f_base = f_base, ok_base = ok_base,
                    num_prompt = num_prompt, model_type = model_type, mnli = True
                )
            else:
                false_scores, true_scores = eval_func(
                    tok, model, prompt_input_group[j], rvs_map[j],
                    t_idx = t_idx, f_idx = f_idx, ok_idx = ok_idx,
                    t_base = t_base, f_base = f_base, ok_base = ok_base,
                    num_prompt = num_prompt, model_type = model_type, mnli = False
                )
            
            if j == 0:
                score_board_0[:, 0] += false_scores # - f_base
                if dataset_name == 'mnli':
                    score_board_0[:, 2] += true_scores # - t_base
                    # score_board_0[:, 1] += ok_scores
                else:
                    score_board_0[:, 1] += true_scores
            if j == 1:
                score_board_1[:, 0] += false_scores # - f_base
                if dataset_name == 'mnli':
                    score_board_1[:, 2] += true_scores # - t_base
                    # score_board_1[:, 1] += ok_scores
                else:
                    score_board_1[:, 1] += true_scores
            
            # '''
            score_board[:, 0] += false_scores # - f_base
            if dataset_name == 'mnli':
                score_board[:, 2] += true_scores # - t_base
                # score_board[:, 1] += ok_scores
            else:
                score_board[:, 1] += true_scores
            # '''

            # print(score_board)
            # abort()
            _, pred = score_board.max(1)
            # print(pred)
            # abort()
        
        _, pred_0 = score_board_0.max(1)
        num_crr_0 += (pred_0.cpu() == label_batch).float().sum()

        _, pred_1 = score_board_1.max(1)
        num_crr_1 += (pred_1.cpu() == label_batch).float().sum()

        _, pred = score_board.max(1)
        num_crr += (pred.cpu() == label_batch).float().sum()
    
    acc_0 = num_crr_0 / num_case
    acc_1 = num_crr_1 / num_case
    acc = num_crr / num_case
    
    print(f'\nAcc_p0 = {acc_0}, Acc_p1 = {acc_1}, Acc = {acc}\n')