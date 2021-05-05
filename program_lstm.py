import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from transformers import GPT2Model, GPT2Tokenizer

from APIs import non_triggers
from l2t_api import APIs, memory_arg_funcs
from l2t_parse_fill import Parser, split, get_col_types


def create_vocab():
    vocab = {'actions': {'nop': 0, 'all_rows': 1}, 'fields': {'no_field': 0}}
    ctr_1, ctr_2 = 2, 1
    for k, v in APIs.items():
        if v['output'] in ['bool', 'any', 'none']:
            continue
        vocab['actions'][k] = ctr_1
        if 'model_args' not in v:
            print(k)
        for ftype in v['model_args']:
            if ftype not in vocab['fields']:
                vocab['fields'][ftype] = ctr_2
                ctr_2 += 1
        ctr_1 += 1
    with open('data/logic_form_vocab.json', 'w') as fp:
        json.dump(vocab, fp, indent=2)
    return


def _get_must_haves(parser, og_sent, table_name, linked_sent):
    raw_sent, tags = parser.normalize(og_sent)
    raw_sent = " ".join(raw_sent)
    sent = parser.initialize_buffer(table_name, linked_sent, tags, raw_sent)[0]
    must_have = []
    for k, v in non_triggers.items():
        if isinstance(v[0], list):
            flags = []
            for v_sub in v:
                flag = False
                for trigger in v_sub:
                    if 'REG:' in trigger:
                        regex = trigger.replace('REG:', '')
                        if re.search(regex, sent):
                            flag = True
                            break
                    elif trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                        if trigger in tags:
                            flag = True
                            break
                    else:
                        if " " + trigger + " " in " " + sent + " ":
                            flag = True
                            break
                flags.append(flag)
            if all(flags):
                must_have.append(k)
        else:
            flag = False
            for trigger in v:
                if 'REG:' in trigger:
                    regex = trigger.replace('REG:', '')
                    if re.search(regex, sent):
                        flag = True
                        break
                elif trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                    if trigger in tags:
                        flag = True
                        break
                else:
                    if " " + trigger + " " in " " + sent + " ":
                        flag = True
                        break
            if flag:
                must_have.append(k)
    return must_have


def inc_precision():
    with open('data/programs.json') as fp:
        data = json.load(fp)

    parser = Parser("data/l2t/all_csv")

    # table_name, og_sent, linked_sent, masked_val, mem_str, mem_num, mem_date, len(c), result
    new_data = []
    for entry in tqdm(data):
        if entry is None:
            continue
        programs = entry[-1]
        if len(programs) == 0:
            continue
        if len(programs) == 1:
            new_data.append(entry)
            continue
        # do trigger-word based filtering to remove false positives
        # og_sent, table_name, linked_sent, raw_sent
        must_have_list = _get_must_haves(parser, entry[1], entry[0], entry[2])

        new_program_list = [p for p in programs if any([f'{mh} {{' in p for mh in must_have_list])]
        if len(new_program_list) == 1:
            new_data.append((*entry[:-1], new_program_list))
            continue

        new_program_list = [p for p in programs if all([f'{mh} {{' in p for mh in must_have_list])]
        if len(new_program_list) == 1:
            new_data.append((*entry[:-1], new_program_list))

        # Do filtering based on used arguments
        # Ideas:
        # 1. Select the ones with most entities
        # 2. Select the ones which rely only on the headers present in the sent except for the generator header
        # 3. Select the ones which use the mask_header as the generator header
        # 4. In case of max { all_rows ; header } and hop { argmax { all_row ; header} ; header } choose first.

        if len(programs) <= 5:
            new_data.append(entry)
        continue

    with open("data/programs_filtered.json", 'w') as f:
        json.dump(new_data, f, indent=2)
    return


class ProgramTree:
    def __init__(self, logic_json, sent=None, table=None):
        self.func = logic_json['func']
        self.args = [ProgramTree(a) if isinstance(a, dict) else a for a in logic_json['args']]
        self.logic_json = logic_json
        self.table = table
        self.sent = ProgramTree.transform_linked_sent(sent, table)
        self._actions = None

    def execute(self):
        pass

    @property
    def action_list(self):
        """
        Returns: The list of actions used to construct the current program tree in dfs order
        """
        if self._actions is not None:
            return self._actions

        def recursive_get(i, logic_json, action_list):
            action_list.append(('func', logic_json['func'], i))
            i = len(action_list) - 1
            for arg in logic_json['args']:
                if isinstance(arg, dict):
                    recursive_get(i, arg, action_list)
                else:
                    if arg != 'all_rows':
                        action_list.append(('tok', arg, i))
                    else:
                        action_list.append(('func', arg, i))
            return

        self._actions = []
        recursive_get(-1, self.logic_json, self._actions)
        return self._actions

    @classmethod
    def from_str(cls, logic_str):
        return ProgramTree(ProgramTree.get_logic_json_from_str(logic_str))

    @staticmethod
    def _sanitize(logic_json):
        new_logic_json = {'func': logic_json['func'], 'args': []}
        if logic_json['func'] in memory_arg_funcs:
            memory_idx = [i for i, a in enumerate(APIs[logic_json['func']]['model_args']) if 'memory' in a][0]
            for a_ix, arg in enumerate(logic_json['args']):
                if a_ix == memory_idx:
                    # print()
                    # print(logic_json['func'], memory_idx)
                    # print(logic_json['args'])
                    new_logic_json['args'].append(f'{arg} ;;; {logic_json["args"][memory_idx + 1]}')
                elif a_ix == memory_idx + 1:
                    continue
                else:
                    if isinstance(arg, dict):
                        new_logic_json['args'].append(ProgramTree._sanitize(arg))
                    else:
                        new_logic_json['args'].append(arg)
            return new_logic_json

        for arg in logic_json['args']:
            if isinstance(arg, dict):
                new_logic_json['args'].append(ProgramTree._sanitize(arg))
            else:
                new_logic_json['args'].append(arg)

        return new_logic_json

    @staticmethod
    def _get_logic_json_from_str(logic_str, curr_buff=''):
        logic_json = {'func': '', 'args': []}
        inside = False
        i = 0
        while i < len(logic_str):
            c = logic_str[i]
            if c == '{':
                if inside:
                    sub_logic_json, i_new = ProgramTree._get_logic_json_from_str(logic_str[i:], curr_buff)
                    logic_json['args'].append(sub_logic_json)
                    curr_buff = ''
                    i += i_new
                    continue
                logic_json['func'] = curr_buff.strip()
                curr_buff = ''
                inside = True
            elif c == ';':
                if curr_buff.strip():
                    logic_json['args'].append(curr_buff.strip())
                curr_buff = ''
            elif c == '}':
                if inside:
                    if curr_buff.strip():
                        logic_json['args'].append(curr_buff.strip())
                    i += 1
                    break
                raise ValueError(f"Misplaced }} at col {i} in str: {logic_str}")
            else:
                curr_buff += c
            i += 1
        return logic_json, i

    @staticmethod
    def get_logic_json_from_str(logic_str):
        return ProgramTree._sanitize(ProgramTree._get_logic_json_from_str(logic_str)[0])

    @staticmethod
    def transform_linked_sent(linked_sent, table):
        """
        e.g. linked sent:
        #philipp petzschner;-1,-1# #partner;0,3# with #jürgen melzer;7,3# for the majority
        of his tennis double tournament .

        transformed sent:
        ^# title ; philipp petzschner #^ ^# type , col , partner #^ with blah blah . The columns are: column1 of type1
        with entry like e, {repeat}...
        """
        if linked_sent is None or table is None:
            return None
        inside = False
        # whether at the index part of the linked entity
        position = False
        position_buf, mention_buf = '', ''
        new_sent = ''
        col2type = get_col_types(table)
        cols = table.columns
        occured_hdrs = set()
        for n in range(len(linked_sent)):
            if linked_sent[n] == '#':
                if position:
                    i = int(split(position_buf, "row"))
                    j = int(split(position_buf, "col"))
                    if i == -1:
                        new_sent += f'[TITLE_START] {mention_buf} [TITLE_END]'
                    elif i == 0:
                        col_type = col2type[j]
                        occured_hdrs.add(j)
                        new_sent += f'[HDR_START] {col_type} ^# {cols[j]} #^ [HDR_END]'
                    else:
                        col_type = col2type[j]
                        new_sent += f'[ENT_START] {col_type} ^# {cols[j]} ;;; {mention_buf} #^ [ENT_END]'

                    # Reset the buffer
                    position_buf = ""
                    mention_buf = ""
                    inside = False
                    position = False
                else:
                    inside = True
            elif linked_sent[n] == ';':
                position = True
            else:
                if position:
                    position_buf += linked_sent[n]
                elif inside:
                    mention_buf += linked_sent[n]
                else:
                    # non-linked words
                    new_sent += linked_sent[n]
        new_sent += ' The other headers in this table are: '
        flag = False
        for j, col in enumerate(cols):
            if j in occured_hdrs:
                continue
            col_type = col2type[j]
            new_sent += f'[HDR_START] {col_type} ^# {col} #^ [HDR_END], '
            flag = True
        if flag:
            new_sent = new_sent[:-2] + ' .'

        new_tokens = []
        inside = False
        for token in new_sent.split(' '):
            if token in ['[ENT_START]', '[ENT_END]', '[HDR_START]', '[HDR_END]', '[TITLE_START]', '[TITLE_END]']:
                if inside:
                    inside = False
                    new_tokens.append(token)
                    continue
                inside = True
                new_tokens.append(token)
                continue
            if inside:
                new_tokens.append(token)
                continue
            pat = r'\d(th|nd|rd)'
            token = token.replace('first', '1st').replace('second', '2nd').replace('third', '3rd').replace(
                'fourth', '4th').replace('fifth', '5th').replace('sixth', '6th').replace('seventh', '7th').replace(
                'eighth', '8th').replace('ninth', '9th').replace('tenth', '10th').replace('eleventh', '11th').replace(
                'twelfth', '12th').replace('thirteenth', '13th').replace('fourteenth', '14th').replace(
                'fifteenth', '15th')
            if len(re.findall(pat, token)) > 0:
                reres = re.findall(r'(\d+)(th|nd|rd)', new_sent)
                if len(reres) == 0:
                    new_tokens.append(token)
                    continue
                # first number in the first matched group
                num = reres[0][0]
                new_tokens.append('[N_START]')
                new_tokens.append('^#')
                new_tokens.append(str(num))
                new_tokens.append('#^')
                new_tokens.append('[N_END]')
            else:
                new_tokens.append(token)
        return ' '.join(new_tokens)

    def __repr__(self):
        return json.dumps(self.logic_json, indent=2)

    def __str__(self):
        return json.dumps(self.logic_json)


class ProgramTreeDataset:
    def __init__(self, programs_trees, vocab, tokenizer=None, cuda=False):
        # take some training files as input
        # This class needs to provide:
        # give the parent-action of another action
        # give the field-type of the action indexed by dfs order. The field type will come from the parent
        self.programs_trees = programs_trees
        self.vocab = vocab
        self.cuda = cuda
        self.max_num_actions = max(len(p.action_list) for p in self.programs_trees)
        self.sent_list = [p.sent for p in self.programs_trees if p.sent is not None]
        if tokenizer is not None:
            self.padded_sequences = tokenizer(self.sent_list, padding=True, truncation=True, return_tensors="pt")
            # of shape (batch_size, seq_len)
            self.input_ids = self.padded_sequences['input_ids']
            self.pad_mask = self.padded_sequences['attention_mask']
            self.tokenizer_dict = {
                'ent_start_tok': tokenizer.convert_tokens_to_ids('[ENT_START]'),
                'ent_end_tok': tokenizer.convert_tokens_to_ids('[ENT_END]'),
                'hdr_start_tok': tokenizer.convert_tokens_to_ids('[HDR_START]'),
                'hdr_end_tok': tokenizer.convert_tokens_to_ids('[HDR_END]'),
                'n_start_tok': tokenizer.convert_tokens_to_ids('[N_START]'),
                'n_end_tok': tokenizer.convert_tokens_to_ids('[N_END]'),
                'fval_start_tok': tokenizer.convert_tokens_to_ids('^#'),
                'fval_end_tok': tokenizer.convert_tokens_to_ids('#^')
            }
            if any(tokenizer.convert_tokens_to_ids(tokenizer.eos_token) == v for v in self.tokenizer_dict.values()):
                raise ValueError(f"Bad tokenizer object with tokenizer_dict {self.tokenizer_dict}")

        assert len(self.sent_list) == 0 or len(self.sent_list) == len(self)

        self._fieldid2field = None

        self.func_idx_matrix, self.func_mask, self.copy_mask = [[] for _ in range(3)]
        total_progs = len(self)
        max_sent_len = 0
        if tokenizer is not None:
            max_sent_len = self.input_ids.size(1)
        self.copy_token_idx_mask = np.zeros((self.max_num_actions, total_progs, max_sent_len), dtype='float32')
        self.generic_copy_mask = np.zeros((self.max_num_actions, total_progs, max_sent_len), dtype='float32')
        if tokenizer is not None:
            self.init_index_tensors(tokenizer)

    def __len__(self):
        return len(self.programs_trees)

    def get_parent_action_ids(self, curr_idx):
        ids = []
        for pt in self.programs_trees:
            if curr_idx < len(pt.action_list):
                parent_action_idx = pt.action_list[curr_idx][-1]
                if parent_action_idx != -1:
                    p_action_name = pt.action_list[parent_action_idx][1]
                    ids.append(self.vocab['actions'][p_action_name])
                else:
                    ids.append(0)
            else:
                ids.append(0)

        return torch.cuda.LongTensor(ids) if self.cuda else torch.LongTensor(ids)

    def get_parent_field_ids(self, curr_idx):
        ids = []
        for pt in self.programs_trees:
            if curr_idx < len(pt.action_list):
                parent_action_idx = pt.action_list[curr_idx][-1]
                if parent_action_idx != -1:
                    p_action_name = pt.action_list[parent_action_idx][1]
                    arg_idx = [ix for ix, a in enumerate(pt.action_list) if a[-1] == parent_action_idx].index(curr_idx)
                    ftype = APIs[p_action_name]['model_args'][arg_idx]
                    ids.append(self.vocab['fields'][ftype])
                else:
                    ids.append(0)
            else:
                ids.append(0)

        return torch.cuda.LongTensor(ids) if self.cuda else torch.LongTensor(ids)

    def _get_field_from_id(self, field_id):
        if self._fieldid2field is None:
            self._fieldid2field = {v: k for k, v in self.vocab['fields'].items()}
        return self._fieldid2field[field_id]

    def init_index_tensors(self, tokenizer):
        def extract1(pt_id, start_tok, end_tok):
            inside1 = False
            inside2 = False
            token_pos_list = []
            for tok_idx, tok in enumerate(self.input_ids[pt_id]):
                tok = tok.item() if isinstance(tok, torch.Tensor) else tok
                if tok == self.tokenizer_dict[start_tok]:
                    inside1 = True
                    continue
                elif tok == self.tokenizer_dict['fval_start_tok'] and inside1:
                    inside2 = True
                    continue
                elif tok == self.tokenizer_dict['fval_end_tok'] and inside1:
                    inside2 = False
                    continue
                elif tok == self.tokenizer_dict[end_tok]:
                    inside1 = False
                    continue
                if not inside2:
                    continue
                token_pos_list.append(tok_idx)
            return token_pos_list

        def extract2(pt_id, tok_list):
            i, j = 0, 0
            token_pos_list = []
            buff_list = []
            while i < len(self.input_ids[pt_id]):
                curr_tok = self.input_ids[pt_id][i]
                curr_tok = curr_tok.item() if isinstance(curr_tok, torch.Tensor) else curr_tok
                if curr_tok == tok_list[j]:
                    buff_list.append(i)
                    i += 1
                    j += 1
                    if len(buff_list) == len(tok_list):
                        token_pos_list.extend(buff_list)
                        buff_list = []
                        j = 0
                else:
                    buff_list = []
                    j = 0
            return token_pos_list

        for curr_ac_ix in range(self.max_num_actions):
            func_idx_row = []
            func_mask_row = []
            copy_mask_row = []
            # of size (batch,)
            parent_field_ids = self.get_parent_field_ids(curr_ac_ix)

            for pt_id, pt in enumerate(self.programs_trees):
                action_idx = action_mask = copy_mask = 0
                if curr_ac_ix < len(pt.action_list):
                    action = pt.action_list[curr_ac_ix][1]
                    action_info = pt.action_list[curr_ac_ix]

                    if action_info[0] == 'func':
                        action_idx = self.vocab['action'][action]
                        action_mask = 1
                    else:
                        # It's a copy token
                        sent = self.sent_list[pt_id]
                        """
                        'ent_start_tok'
                        'ent_end_tok'
                        'hdr_start_tok'
                        'hdr_end_tok'
                        'n_start_tok'
                        'n_end_tok'
                        'fval_start_tok'
                        'fval_end_tok'
                        """
                        field_type = self._get_field_from_id(parent_field_ids[pt_id])
                        if field_type == 'n':
                            tok_pos_list = extract1(pt_id, 'n_start_tok', 'n_end_tok')
                        elif 'header' in field_type:
                            tok_pos_list = extract1(pt_id, 'hdr_start_tok', 'hdr_end_tok')
                        elif 'memory' in field_type:
                            tok_pos_list = extract1(pt_id, 'ent_start_tok', 'ent_end_tok')
                        else:
                            raise ValueError(f"Action {action_info} doesn't match field type {field_type}")
                        self.generic_copy_mask[curr_ac_ix, pt_id, tok_pos_list] = 1.

                        copy_utterance = '^# ' + action_info[1] + ' #^'
                        copy_utterance_toks = tokenizer.encode(copy_utterance)
                        tok_pos_list = extract2(pt_id, copy_utterance_toks)[1:-1]
                        self.copy_token_idx_mask[curr_ac_ix, pt_id, tok_pos_list] = 1.
                        copy_mask = 1

                func_idx_row.append(action_idx)
                func_mask_row.append(action_mask)
                copy_mask_row.append(copy_mask)

            self.func_idx_matrix.append(func_idx_row)
            self.func_mask.append(func_mask_row)
            self.copy_mask.append(copy_mask_row)

        T = torch.cuda if self.cuda else torch
        self.func_idx_matrix = T.LongTensor(self.func_idx_matrix)
        self.func_mask = T.FloatTensor(self.func_mask)
        self.copy_mask = T.FloatTensor(self.copy_mask)
        self.copy_token_idx_mask = torch.from_numpy(self.copy_token_idx_mask)
        self.generic_copy_mask = torch.from_numpy(self.generic_copy_mask)
        if self.cuda:
            self.copy_token_idx_mask = self.copy_token_idx_mask.cuda()
            self.generic_copy_mask = self.generic_copy_mask.cuda()
        return


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, encoder_hidden_size):
        super(PointerNet, self).__init__()
        self.proj_linear = nn.Linear(encoder_hidden_size, query_vec_size, bias=False)

    def forward(self, sent_encodings, copy_token_mask, query_vec):
        """
        :param sent_encodings: (batch_size, seq_len, encoder_hidden_size)
        :param copy_token_mask: (action_seq_len, batch_size, seq_len) contains bool which indicates the positions to potentially copy
        :param query_vec: (action_seq_len, batch_size, attn_vector_size)
        :return: (action_seq_len, batch_size, seq_len) copy probs
        """
        sent_encodings = self.proj_linear(sent_encodings)
        # (batch_size, 1, seq_len, attn_vector_size)
        sent_encodings = sent_encodings.unsqueeze(1)

        # (batch_size, action_seq_len, attn_vector_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        # (batch_size, action_seq_len, seq_len)
        weights = torch.matmul(sent_encodings, q).squeeze(3)

        # (action_seq_len, batch_size, seq_len)
        weights = weights.permute(1, 0, 2)

        # (action_seq_len, batch_size, seq_len)
        weights.data.masked_fill_(copy_token_mask, -float('inf'))

        # (action_seq_len, batch_size, seq_len)
        ptr_weights = F.softmax(weights, dim=-1)
        return ptr_weights


class ProgramLSTM(nn.Module):
    """
    Takes as input a linked sentence and constructs a logic form AST recursively (left-to-right dfs).
    """

    def __init__(self, action_embed_size, field_embed_size, decoder_hidden_size,
                 attn_vec_size, dropout, device='cpu', gpt_model='gpt2'):
        super(ProgramLSTM, self).__init__()
        self.action_embed_size = action_embed_size
        self.attn_vec_size = attn_vec_size
        self.field_embed_size = field_embed_size
        self.decoder_hidden_size = decoder_hidden_size
        self.gpt_model = gpt_model
        self.device = torch.device(device) if isinstance(device, str) else device
        with open('data/logic_form_vocab.json') as fp:
            self.vocab = json.load(fp)

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model, padding_side='left')
        self.tokenizer.add_tokens(['[ENT_START]', '[ENT_END]', '[HDR_START]', '[HDR_END]',
                                   '^#', '#^', '[TITLE_START]', '[TITLE_END]', '[N_START]', '[N_END]'])
        if 'gpt2' in gpt_model:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = GPT2Model.from_pretrained(gpt_model)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        # encoder_hidden_size = self.encoder.config.hidden_size
        encoder_hidden_size = self.encoder.config.n_embd
        self.encoder.to(device)

        # embedding for funcs
        self.action_embed = nn.Embedding(len(self.vocab['actions']), action_embed_size)
        # embedding for field types
        self.field_embed = nn.Embedding(len(self.vocab['fields']), field_embed_size)

        input_dim = action_embed_size  # previous action
        input_dim += attn_vec_size  # size of attentional hidden state
        # Parent feeding
        input_dim += action_embed_size  # parent action
        input_dim += field_embed_size  # parent field embed
        input_dim += decoder_hidden_size  # parent's decoder state

        self.decoder_lstm = nn.LSTMCell(input_dim, decoder_hidden_size)
        self.decoder_cell_init = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        # Pointer net for copying tokens from input seq
        self.pointer_net = PointerNet(attn_vec_size, encoder_hidden_size)

        # Project encoder hidden state to decoder hidden state dim
        self.attn_1_linear = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector/attention hidden state` in (Luong et al., 2015)
        # The attentional vector is then fed through the softmax layer to produce the predictive distribution
        # h~t = tanh(Wc[ct ;ht])
        self.attn_2_linear = nn.Linear(decoder_hidden_size + encoder_hidden_size, attn_vec_size, bias=False)

        # We feed the attentional vector (i.e., the query vector) into a linear layer without bias, and
        # compute action probabilities by dot-producting the resulting vector and
        # (Func, all_rows, CopyToken) action embeddings
        # i.e., p(action) = query_vec^T \cdot W \cdot embedding
        self.query_vec_to_action_embed = nn.Linear(attn_vec_size, action_embed_size, bias=False)
        self.action_readout_b = nn.Parameter(torch.FloatTensor(len(self.vocab['actions'])).zero_())
        self.action_readout = lambda q: F.linear(
            self.query_vec_to_action_embed(q), self.action_embed.weight, self.action_readout_b)

        self.dropout = nn.Dropout(dropout)

        if device != 'cpu':
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor
        return

    @staticmethod
    def dot_prod_attention(h_t, sent_encoding, transformed_sent_encodings, mask=None):
        """
        :param h_t: curr_hidden_state - (batch_size, decoder_hidden_size)
        :param sent_encoding: (batch_size, seq_len, encoder_hidden_size)
        :param transformed_sent_encodings: (batch_size, seq_len, decoder_hidden_size)
        :param mask: (batch_size, seq_len)
        """
        # attn_weight - (batch_size, seq_len)
        # bmm multiplies two tensors: b,m,p with b,p,n to get b,m,n
        attn_weight = torch.bmm(transformed_sent_encodings, h_t.unsqueeze(2)).squeeze(2)
        if mask is not None:
            attn_weight.data.masked_fill_(mask, -float('inf'))
        # softmax over last dim of seq_len size
        attn_weight = F.softmax(attn_weight, dim=-1)

        # (batch_size, 1, seq_len)
        att_view = attn_weight.size(0), 1, attn_weight.size(1)
        # (batch_size, encoder_hidden_size)
        ctx_vec = torch.bmm(attn_weight.view(*att_view), sent_encoding).squeeze(1)
        return ctx_vec

    def encode(self, padded_sequences):
        """
        Tokenizes and encodes the sent_list and outputs the hidden states for each token

        Args:
            padded_sequences: dict containing input_ids & attention mask of shape (batch_size, max_seq_len/seq_len)

        Returns: hidden_states - (batch_size, seq_len, encoder_hidden_size), padding_mask - (batch_size, seq_len)

        """
        # padded_sequences = self.tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt")
        encoder_output = self.encoder(**padded_sequences)
        # (batch_size, seq_len, encoder_hidden_size)
        hidden_states = encoder_output.last_hidden_state
        # (batch_size, seq_len)
        mask = padded_sequences['attention_mask']
        return hidden_states, mask

    def init_decoder_state(self, enc_last_state):
        """
        Compute the initial decoder hidden state and cell state from the last state of encoder

        Args:
            enc_last_state: of shape (batch_size, encoder_hidden_size)

        Returns: (batch_size, decoder_hidden_size) h_0, (batch_size, decoder_hidden_size) c_0

        """
        h_0 = self.decoder_cell_init(enc_last_state)
        return h_0, self.new_tensor(h_0.size()).zero_()

    def decoder_step(self, x, hc_pair, sent_encodings, transformed_sent_encodings,
                     attn_token_mask=None):
        """Perform a single time-step of computation in decoder LSTM

        Args:
            x: (batch_size, decoder_input_dim) input
            hc_pair: Tuple[(batch_size, decoder_hidden_size), (batch_size, decoder_hidden_size)], previous hidden and cell state
            sent_encodings: (batch_size, seq_len, encoder_hidden_size) sentence encodings
            transformed_sent_encodings: (batch_size, seq_len, decoder_hidden_size) linearly transformed sent encodings for dot product with curr hidden state
            attn_token_mask: (batch_size, seq_len) mask over source tokens (Note: unused entries are masked to **one**)

        Returns:
            The new LSTM hidden state, cell state and attentional hidden state
        """
        # (batch, decoder_hidden_size) - ht, ct
        h_t, cell_t = self.decoder_lstm(x, hc_pair)

        # ctx_t - (batch_size, encoder_hidden_size)
        ctx_t = ProgramLSTM.dot_prod_attention(
            h_t, sent_encodings, transformed_sent_encodings, mask=attn_token_mask)

        # h~t = tanh(Wc[ct ;ht])
        # attn_t - (batch_size, attn_vector_size)
        attn_t = torch.tanh(self.attn_2_linear(torch.cat([h_t, ctx_t], 1)))
        attn_t = self.dropout(attn_t)

        return (h_t, cell_t), attn_t

    def decode(self, batch, sent_encodings, dec_init_vec, attn_token_mask):
        """Given a batch of examples and their encodings of input utterances, compute query vectors at each decoding
         time step, which are used to compute action probabilities

        Args:
            batch: a `ProgramTreeDataset` object storing input examples
            sent_encodings: of shape (batch_size, seq_len, encoder_hidden_size)
            dec_init_vec: (h_0, c_0) for decoder lstm of shape (batch_size, encoder_hidden_size)
            attn_token_mask: of shape (batch_size, seq_len) used to mask attention on pad tokens

        Returns:
            Query vectors/attentional vectors/attentional hidden state, of shape (action_seq_len, batch_size, decoder_hidden_size)
        """
        batch_size = len(batch)
        hc_pair = dec_init_vec
        transformed_sent_encodings = self.attn_1_linear(sent_encodings)
        zero_action_embed = self.new_tensor(self.action_embed_size).zero_()

        attn_vecs = []
        history_states = []

        for t in range(batch.max_num_actions):
            # the input to the decoder LSTM is a concatenation of multiple signals
            # [
            #   embedding of previous action -> `prev_action_embed`,
            #   previous attentional vector/hidden state -> `prev_attn_t`,
            #   embedding of the parent's action -> `parent_action_embed`,
            #   embedding of the parent field -> `parent_field_embed`,
            #   LSTM hidden state which (eventually) produced parent action -> `parent_states`
            # ]
            if t == 0:
                x = self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_()
                # offset = self.action_embed_size
                # offset += self.attn_vec_size
                # offset += self.action_embed_size
                # offset += self.field_embed_size
            else:
                prev_action_embeds = []
                for program_tree in batch.program_trees:
                    # action t - 1
                    if t < len(program_tree.action_list):
                        prev_action = program_tree.action_list[t - 1]
                        if prev_action[0] == 'func':
                            prev_action_embed = self.action_embed.weight[self.vocab['actions'][prev_action[1]]]
                        else:
                            raise ValueError(f"Wrong previous action type: {prev_action} in tree {program_tree}")
                    else:
                        prev_action_embed = zero_action_embed

                    prev_action_embeds.append(prev_action_embed)

                # (batch_size, action_embed_size)
                prev_action_embeds = torch.stack(prev_action_embeds)

                # noinspection PyUnboundLocalVariable
                inputs = [prev_action_embeds, prev_attn_t,
                          self.action_embed(batch.get_parent_action_ids(t)),  # parent action embed
                          self.field_embed(batch.get_parent_field_ids(t)),  # parent field embed
                          ]

                # append history states
                actions_t = [pt.action_list[t] if t < len(pt.action_list) else None for pt in batch.program_trees]
                parent_states = torch.stack([history_states[p_t][0][batch_id]
                                             for batch_id, p_t in
                                             enumerate(a_t[-1] if a_t else 0 for a_t in actions_t)])

                inputs.append(parent_states)
                x = torch.cat(inputs, dim=-1)

            # attn_t - (batch_size, attn_vector_size)
            hc_pair, attn_t = self.decoder_step(
                x, hc_pair, sent_encodings, transformed_sent_encodings, attn_token_mask=attn_token_mask)

            history_states.append(hc_pair)
            attn_vecs.append(attn_t)
            prev_attn_t = attn_t

        # attn_vecs - (action_seq_len, batch_size, attn_vector_size)
        attn_vecs = torch.stack(attn_vecs, dim=0)
        return attn_vecs

    def score(self, program_trees):
        """Given a list of examples, compute the log-likelihood of generating the target AST

        Args:
            program_trees: a batch of program trees
        output: score for each training example of shape (batch_size,)
        """
        # batch index vector summary:
        # 1. func_idx_matrix (action_seq_len, batch_size) stores the target action ids
        # 2. func_mask (action_seq_len, batch_size) stores a bool indicating if the action is of 'func' type
        # 3. copy_mask (action_seq_len, batch_size) is equal to negation of func_mask
        # 4. copy_token_idx_mask (action_seq_len, batch_size, seq_len) stores
        #    a bool indicating which tokens should be copied
        batch = ProgramTreeDataset(program_trees, self.vocab, self.tokenizer, cuda=self.device != 'cpu')
        sent_encodings, pad_masks = self.encode(batch.padded_sequences)
        dec_init_vec = self.init_decoder_state(sent_encodings[:, -1, :])
        # query_vectors are attention hidden states h_t~ of the decoder
        # shape - (action_seq_len, batch_size, attn_vector_size)
        query_vectors = self.decode(batch, sent_encodings, dec_init_vec, pad_masks)

        # shape - (action_seq_len, batch_size, num_actions)
        apply_func_prob = F.softmax(self.action_readout(query_vectors), dim=-1)

        # shape - (action_seq_len, batch_size)
        target_apply_func_prob = torch.gather(apply_func_prob, dim=2,
                                              index=batch.func_idx_matrix.unsqueeze(2)).squeeze(2)

        # pointer network copying scores over source tokens
        # (action_seq_len, batch_size, src_sent_len)
        copy_prob = self.pointer_net(sent_encodings, batch.generic_copy_mask, query_vectors)

        # marginalize over the copy probabilities that generate the correct token
        # (action_seq_len, batch_size)
        target_copy_prob = torch.sum(copy_prob * batch.copy_token_idx_mask, dim=-1)

        # (action_seq_len, batch_size)
        action_prob = target_apply_func_prob * batch.func_mask + target_copy_prob * batch.copy_mask
        action_prob = action_prob.log()
        scores = torch.sum(action_prob, dim=0)
        return scores

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            'args': (self.action_embed_size, self.field_embed_size,
                     self.decoder_hidden_size, self.attn_vec_size,
                     self.dropout, self.gpt_model),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
        return

    @classmethod
    def load(cls, model_path, cuda=False):
        device = 'cuda' if cuda else 'cpu'
        params = torch.load(model_path, map_location=torch.device(device))
        (action_embed_size, field_embed_size, decoder_hidden_size,
         attn_vec_size, dropout, gpt_model) = params['args']

        plstm = cls(action_embed_size, field_embed_size, decoder_hidden_size,
                    attn_vec_size, dropout, device, gpt_model)
        saved_state = params['state_dict']
        plstm.load_state_dict(saved_state)
        if cuda:
            plstm = plstm.cuda()
        plstm.eval()
        return plstm


def test_program_tree():
    # l_str = "hop { nth_argmax { all_rows ; attendance ; 2 } ; home team }=milton keynes dons/True"
    # a1 = ProgramTree.from_str(l_str)
    # print(repr(a1))
    # print(a1.action_list)
    #
    # l_str = ("and { eq { nth_min { all_rows ; round ; 3 } ; 3 } ;"
    #          " eq { hop { nth_argmin { all_rows ; round ; 3 } ; name } ; scott starks } } = true")
    # a2 = ProgramTree.from_str(l_str)
    # print(repr(a2))
    # print(a2.action_list)
    #
    # l_str = ("and { "
    #          "only { filter_greater { filter_eq { all_rows ; decision ; labarbera } ; attendance ; 18000 } } ;"
    #          " eq { hop { "
    #          "filter_greater { filter_eq { all_rows ; decision ; labarbera } ; attendance ; 18000 } ;"
    #          " date } ; november 3 } } = true")
    # a3 = ProgramTree.from_str(l_str)
    # print(repr(a3))
    # print(a3.action_list)

    l_str = "greater_str_inv { all_rows ; venue ; belk gymnasium ; closed }=charlotte speedway/True"
    a4 = ProgramTree.from_str(l_str)
    print(repr(a4))
    print(a4.action_list)

    with open('data/logic_form_vocab.json') as fp:
        vocab = json.load(fp)
    # b = ProgramTreeDataset([a1, a2, a3, a4], vocab)
    b = ProgramTreeDataset([a4], vocab)
    print(b.get_parent_action_ids(5))
    print(b.get_parent_field_ids(5))

    ns = ProgramTree.transform_linked_sent(
        '#sap g33k;5,2# be the team with the third highest #team number;0,3# in the #first championship;-1,-1# .',
        pd.read_csv('data/l2t/all_csv/2-15584199-3.html.csv', delimiter="#")
    )
    print(ns)

    l_str = "hop { nth_argmax { all_rows ; total ; 2 } ; province }=chonburi/True"
    a5 = ProgramTree.from_str(l_str)
    print(repr(a5))
    print(a5.action_list)

    with open('data/logic_form_vocab.json') as fp:
        vocab = json.load(fp)
    b = ProgramTreeDataset([a5], vocab)
    print(b.get_parent_action_ids(5))
    print(b.get_parent_field_ids(5))

    ns = ProgramTree.transform_linked_sent(
        '#chonburi;2,1# receive the 2nd highest #total;0,5# medal count in the #2008 thailand national game;-1,-1# .',
        pd.read_csv('data/l2t/all_csv/2-14892957-1.html.csv', delimiter="#")
    )
    print(ns)
    return


def tmp_test():
    parser = Parser("data/l2t/all_csv")
    must_have_list = _get_must_haves(
        parser,
        'most of the hit \'n run tour concert cities were in the united states .',
        '2-12946465-1.html.csv',
        'most of the #hit \' n run tour;-1,-1# concert #city;0,1# be in the #united states;6,2# .')
    print(must_have_list)
    return


if __name__ == '__main__':
    # create_vocab()
    test_program_tree()
    # inc_precision()
    # tmp_test()