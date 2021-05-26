import argparse
import copy
import json
# noinspection PyUnresolvedReferences
import pprint
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# noinspection PyUnresolvedReferences
from torch import nn, optim, autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import GPT2Model, GPT2Tokenizer, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, AdamW, \
    get_linear_schedule_with_warmup

from APIs import non_triggers
from l2t_api import APIs, memory_arg_funcs, check_if_accept, sharper_triggers, pat_month, pat_year, pat_day, month_map
from l2t_parse_fill import Parser, split, get_col_types
from utils import tile

NEW_TOKENS = ['hfuewjlr', 'cotbwpry', 'gyhulcem', 'uzdfpzvk', 'gifyoazr', 'ogvrhlel', 'hrcdtosp', 'yvyzclyh',
              'nvoqnztx', 'zfjxetwn', 'rioxievv', 'ccfriagn', 'nqhuopoc', 'huombchu', 'udpvyfhn', 'kjjyzupm',
              'dmpfjonu', 'bwzxwcoa', 'iezmaqxb', 'qidywllz', 'glrximum', 'cqwpuiux', 'zxlwfmab', 'bcixyahe',
              'vuxnzyfm', 'taynudla', 'vmxlasbt', 'fpvzuurn', 'srdstkko', 'bytzjzbf', 'zwwszhfu', 'viyhhwec',
              'uzmtiymv', 'wdncdeqw', 'vdkrbghd',

              '[MASK]', '[ENT_START]', '[ENT_END]', '[HDR_START]', '[HDR_END]',
              '^#', '#^', '[TITLE_START]', '[TITLE_END]', '[N_START]', '[N_END]'
              ]


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


def _get_must_haves(parser, og_sent, table_name, linked_sent, trigger_type='soft'):
    raw_sent, tags = parser.normalize(og_sent)
    raw_sent = " ".join(raw_sent)
    sent = parser.initialize_buffer(table_name, linked_sent, tags, raw_sent)[0]
    must_have = []
    triggers = non_triggers if trigger_type == 'soft' else sharper_triggers
    for k, v in triggers.items():
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


def _compare(_a, _b):
    _a = str(_a)
    _b = str(_b)
    if _a == _b:
        return True
    try:
        _a = float(_a)
        _b = float(_b)
        return _a == _b
    except:
        return False


def _date_compare(actual, returned):
    if len(re.findall(pat_month, actual)) > 0:
        year_val1 = re.findall(pat_year, actual)
        year_val2 = re.findall(pat_year, returned)
        b1, b2 = len(year_val1) == 0, len(year_val2) == 0
        if (b1 and not b2) or (not b1 and b2):
            return False

        month_val1 = re.findall(pat_month, actual)
        month_val2 = re.findall(pat_month, returned)
        b1, b2 = len(month_val1) == 0, len(month_val2) == 0
        if (b1 and not b2) or (not b1 and b2):
            return False

        day_val1 = re.findall(pat_day, actual)
        day_val2 = re.findall(pat_day, returned)
        b1, b2 = len(day_val1) == 0, len(day_val2) == 0
        if (b1 and not b2) or (not b1 and b2):
            return False

        if len(year_val1) == 0:
            year_val1 = year_val2 = int("2260")
        else:
            year_val1 = int(year_val1[0])
            year_val2 = int(year_val2[0])

        if len(day_val1) == 0:
            day_val1 = day_val2 = int("1")
        else:
            day_val1 = int(day_val1[0])
            day_val2 = int(day_val2[0])

        if len(month_val1) == 0:
            month_val1 = month_val2 = int("1")
        else:
            month_val1 = month_map[month_val1[0]]
            month_val2 = month_map[month_val2[0]]

        return (year_val1, month_val1, day_val1) == (year_val2, month_val2, day_val2)
    return False


def inc_precision(thresh=3, test_sent=''):
    with open('data/programs.json') as fp:
        data = json.load(fp)

    if len(test_sent) > 0:
        idx = [i for i, ent in enumerate(data) if ent is not None and ent[1] == test_sent]
        data = [data[i] for i in idx]

    parser = Parser("data/l2t/all_csv")

    def get_all_headers(logic_json):
        func = logic_json['func']
        all_headers = []
        if func in APIs:
            all_hdr_args = [aidx for aidx, a in enumerate(APIs[func]['argument']) if 'header' in a]
            if len(all_hdr_args) > 0:
                fargs = [arg.split(';;;') if isinstance(arg, str) else [arg] for arg in logic_json['args']]
                fargs = [y.strip() if isinstance(y, str) else y for x in fargs for y in x]
                all_headers.extend(fargs[i] for i in all_hdr_args)
        for arg in logic_json['args']:
            if isinstance(arg, dict):
                all_headers.extend(get_all_headers(arg))
        return all_headers

    def get_new_entry(en, cl, c2t):
        masked_sent, mapping = parser.fake_parse(en[0], en[1])
        ts = ProgramTree.transform_linked_sent(masked_sent, mapping, cl, c2t, en[3])
        return *en[:3], ts, *en[3:7], cl, c2t, en[-1]

    def get_linked_hdrs(linked_sent, cols):
        inside = False
        position = False
        position_buf, mention_buf = '', ''
        new_sent = ''
        occured_hdrs = set()
        for n in range(len(linked_sent)):
            if linked_sent[n] == '#':
                if position:
                    i = int(split(position_buf, "row"))
                    j = int(split(position_buf, "col"))
                    if j != -1:
                        occured_hdrs.add(cols[j])

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
        return occured_hdrs

    # table_name, og_sent, linked_sent, masked_val, mem_str, mem_num, mem_date, len(c), result
    new_data = []
    for entry in tqdm(data):
        if entry is None:
            continue
        all_programs_list = []
        programs = entry[-1]
        table = pd.read_csv(f'data/l2t/all_csv/{entry[0]}', delimiter="#")
        col2type = get_col_types(table)
        cols = table.columns.tolist()
        if len(programs) == 0:
            continue
        if len(programs) == 1:
            new_data.append(get_new_entry(entry, cols, col2type))
            continue
        # do trigger-word based filtering to remove false positives
        must_have_list = _get_must_haves(parser, entry[1], entry[0], entry[2])
        new_program_list = [p_ix for p_ix, p in enumerate(programs) if any([f'{mh} {{' in p for mh in must_have_list])]
        all_programs_list.append(new_program_list)

        # do a more intense trigger based filtering to differentiate 'more' from 'less'
        must_have_list = _get_must_haves(parser, entry[1], entry[0], entry[2], trigger_type='hard')
        new_program_list = [p_ix for p_ix, p in enumerate(programs) if any([f'{mh} {{' in p for mh in must_have_list])]
        all_programs_list.append(new_program_list)

        # Do filtering based on used arguments
        # Ideas:
        # 1. Select the ones with most entities
        # 2. Select the ones which rely only on the headers present in the sent except for the generator header
        # 3. Select the ones which use the mask_header as the generator header
        # 4. In case of max { all_rows ; header } and hop { argmax { all_row ; header} ; header } choose first.
        # 5. Select the ones which generate the full str output instead of just the int part
        # 6. Select the ones which rely on nth_arg if present
        # 7. If the mask matches pat_month, reject _num programs
        # 8. Out of all date generating programs, choose the most specific one
        program_trees = [ProgramTree.from_str(p_str) for p_str in programs]
        msk_header = entry[3][0][4:].strip()
        if msk_header == 'input':
            # select the count funcs
            new_program_list = [p_ix for p_ix, p in enumerate(programs) if 'count {' in p]
            all_programs_list.append(new_program_list)
        else:
            # Select the func which has the correct generating header
            new_program_list = []
            for pt_idx, pt in enumerate(program_trees):
                lj = pt.logic_json
                func, fargs = lj['func'], lj['args']
                fargs = [arg.split(';;;') if isinstance(arg, str) else [arg] for arg in fargs]
                fargs = [y.strip() if isinstance(y, str) else y for x in fargs for y in x]
                all_hdr_args = [aidx for aidx, a in enumerate(APIs[func]['argument']) if 'header' in a]
                if len(all_hdr_args) > 0:
                    harg_idx = all_hdr_args[0]
                    hval = fargs[harg_idx]
                    if hval == msk_header:
                        new_program_list.append(pt_idx)
            all_programs_list.append(new_program_list)

        # Select the ones which rely only on linked headers
        new_program_list = []
        for pt_idx, pt in enumerate(program_trees):
            all_used_hdrs = set(get_all_headers(pt.logic_json))
            linked_hdrs = get_linked_hdrs(entry[2], cols)
            if len(all_used_hdrs - linked_hdrs) == 0:
                new_program_list.append(pt_idx)
        all_programs_list.append(new_program_list)

        # select the ones which use most linked headers
        prog2num = {}
        for pt_idx, pt in enumerate(program_trees):
            all_used_hdrs = set(get_all_headers(pt.logic_json))
            linked_hdrs = get_linked_hdrs(entry[2], cols)
            prog2num[pt_idx] = len(linked_hdrs.intersection(all_used_hdrs))
        max_num = max(prog2num.values())
        new_program_list = [pt_idx for pt_idx, num in prog2num.items() if num == max_num]
        all_programs_list.append(new_program_list)

        # Select the ones which generate the full str
        new_program_list = []
        for p_ix, prog in enumerate(programs):
            ret_val = prog[prog.rfind('=') + 1:-5]
            if _compare(ret_val, entry[3][1]):
                new_program_list.append(p_ix)
        all_programs_list.append(new_program_list)

        # Select the ones which rely on nth_arg if present
        if any(msk_num for msk_num in entry[5] if msk_num[0] == 'ntharg'):
            ordinal_funcs = ['nth_max {', 'nth_argmin {', 'nth_argmax {', 'nth_min {']
            new_program_list = [p_ix for p_ix, p in enumerate(programs) if any(f in p for f in ordinal_funcs)]
            all_programs_list.append(new_program_list)

        if len(re.findall(pat_month, str(entry[3][1]))) > 0:
            new_program_list = []
            for p_ix, prog in enumerate(programs):
                if '_date {' not in prog:
                    continue
                ret_val = prog[prog.rfind('=') + 1:-5]
                if _date_compare(str(entry[3][1]), ret_val):
                    new_program_list.append(p_ix)
            all_programs_list.append(new_program_list)

        # filter based on count
        pix_ctr = Counter(y for x in all_programs_list for y in x)
        if len(pix_ctr) == 0 and len(programs) <= thresh:
            new_data.append(get_new_entry(entry, cols, col2type))
            continue
        elif len(pix_ctr) == 0:
            continue
        max_count = max(pix_ctr.values())
        chosen_pix_list = [k for k, v in pix_ctr.items() if v == max_count]
        new_program_list = [programs[pix] for pix in chosen_pix_list]
        if len(new_program_list) <= thresh:
            new_data.append((*get_new_entry(entry, cols, col2type)[:-1], new_program_list))
        # if len(programs) <= 5:
        #     new_data.append(get_new_entry(entry, cols, col2type))
        continue

    if len(test_sent) == 0:
        with open("data/programs_filtered.json", 'w') as f:
            json.dump(new_data, f, indent=2)
    else:
        pprint.pprint(new_data, indent=2)
    return


def get_val(pos_tensor):
    return pos_tensor.item() if isinstance(pos_tensor, torch.Tensor) else pos_tensor


class ProgramTree:
    def __init__(self, logic_json, masked_ls=None, mapping=None, cols=None, col2type=None, masked_val=None):
        self.func = logic_json['func']
        self.logic_json = logic_json
        self.sent = ProgramTree.transform_linked_sent(masked_ls, mapping, cols, col2type, masked_val)
        self._actions = None

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
    def from_str(cls, logic_str, masked_ls=None, mapping=None, cols=None, col2type=None, masked_val=None):
        return cls(cls.get_logic_json_from_str(logic_str), masked_ls, mapping, cols, col2type, masked_val)

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
    def transform_linked_sent(masked_ls, mapping, cols, col2type, masked_val=None):
        """
        e.g. linked sent:
        <TITLE> have a higher <ENTITY1> in <ENTITY2> than he do in <ENITTY3> .

        transformed sent:
        [TITLE_START] mark donohue [TITLE_END] have a higher [HDR_START] num ^# start #^ hfuewjlr [HDR_END] in [MASK]
         than he do in [ENT_START] num ^# year ;;; 1970 #^ cotbwpry [ENT_END] .
         The other headers in this table are: [HDR_START] num ^# year #^ gyhulcem [HDR_END] ,
          [HDR_START] str ^# chassis #^ uzdfpzvk [HDR_END] , [HDR_START] str ^# engine #^ gifyoazr [HDR_END] ,
          [HDR_START] num ^# finish #^ ogvrhlel [HDR_END] , [HDR_START] str ^# entrant #^ hrcdtosp [HDR_END]  .
        """

        if any(x is None for x in [masked_ls, mapping, cols, col2type]):
            return None
        new_sent = []
        tag_ctr = 0
        all_types = {'<ENTITY', '<NONENTITY', '<NARG', '<COMPUTE', '<NONLINKED', '<COUNT'}

        occured_hdrs = set()
        for tok in masked_ls.split():
            if tok[0] != '<' or tok[-1] != '>' or not any(t in tok for t in all_types):
                new_sent.append(tok)
                continue
            v, (i, j) = mapping[tok]
            # -5 nonlink, -4 count, -3 compute, -7 nth, -1 title
            if i in [-5, -4]:
                comp_msk_val_0 = 'msk_nonlink_num' if i == -5 else 'msk_input'
                if masked_val is not None and masked_val[0] == comp_msk_val_0 and _compare(v, masked_val[1]):
                    new_sent.append('[MASK]')
                else:
                    new_sent.append(str(v))
            elif i == -3 and j == -3:
                if masked_val is not None and _compare(v, str(masked_val[1])):
                    new_sent.append('[MASK]')
                else:
                    new_sent.append(str(v))
            elif i == -7:
                new_sent.append(f'[N_START] ^# {v} #^ {NEW_TOKENS[tag_ctr]} [N_END]')
                tag_ctr += 1
                tag_ctr %= len(NEW_TOKENS)
            elif i == -1:
                new_sent.append(f'[TITLE_START] {v} [TITLE_END]')
            elif i == 0:
                col_type = col2type[j]
                occured_hdrs.add(j)
                new_sent.append(f'[HDR_START] {col_type} ^# {cols[j]} #^ {NEW_TOKENS[tag_ctr]} [HDR_END]')
                tag_ctr += 1
                tag_ctr %= len(NEW_TOKENS)
            else:
                # -3, normal
                col_type = col2type[j]
                if masked_val is not None and cols[j] == masked_val[0][4:] and _compare(v, str(masked_val[1])):
                    new_sent.append('[MASK]')
                else:
                    new_sent.append(f'[ENT_START] {col_type} ^# {cols[j]} ;;; {v} #^'
                                    f' {NEW_TOKENS[tag_ctr]} [ENT_END]')
                    tag_ctr += 1
                    tag_ctr %= len(NEW_TOKENS)

        new_sent = ' '.join(new_sent)
        new_sent += ' The other headers in this table are: '
        flag = False
        for j, col in enumerate(cols):
            if j in occured_hdrs:
                continue
            col_type = col2type[j]
            new_sent += f'[HDR_START] {col_type} ^# {col} #^ {NEW_TOKENS[tag_ctr]} [HDR_END] , '
            tag_ctr += 1
            tag_ctr %= len(NEW_TOKENS)
            flag = True
        if flag:
            new_sent = new_sent[:-2] + ' .'

        return new_sent

    @staticmethod
    def get_logic_json_from_action_list(model_action_list, trans_link_sent):
        """
        Converts an action list received from inference to the logic_json dict with the help of transformed linked sent
        Args:
            model_action_list:
            trans_link_sent:

        Returns: logic_json dict
        """
        tok2ent = {}
        inside = False
        buf = []
        tokenized_sent = trans_link_sent.split()
        for tok_idx, tok in enumerate(tokenized_sent):
            if tok == '^#':
                inside = True
                continue
            if not inside:
                continue
            if tok == '#^':
                tok2ent[tokenized_sent[tok_idx + 1]] = ' '.join(buf)
                buf = []
                inside = False
            else:
                buf.append(tok)
        action_list = []
        for act in model_action_list:
            if act[0] == 'func':
                action_list.append(act)
            else:
                if act[1] in tok2ent:
                    mapped_ent = tok2ent[act[1]]
                else:
                    raise ValueError(f"Bad parameters: {model_action_list}\n{trans_link_sent}")
                action_list.append(('tok', mapped_ent))

        logic_json = {'func': action_list[0][1], 'args': []}
        curr_args_stack = [logic_json['args']]
        curr_func_stack = [action_list[0][1]]
        for act in action_list[1:]:
            if act[0] == 'func' and act[1] != 'all_rows':
                sub_lj = {'func': act[1], 'args': []}
                curr_args_stack[-1].append(sub_lj)
                curr_func_stack.append(act[1])
                curr_args_stack.append(sub_lj['args'])
            else:
                if ';;;' in act[1]:
                    a = act[1].split(';;;')
                    a = [x.strip() for x in a]
                    curr_args_stack[-1].extend(a)
                else:
                    curr_args_stack[-1].append(act[1])
            if len(curr_args_stack[-1]) == len(APIs[curr_func_stack[-1]]['argument']):
                curr_func_stack.pop()
                curr_args_stack.pop()
            if len(curr_func_stack) == 0:
                break
        return logic_json

    @staticmethod
    def execute(table_name, logic_json, table=None):
        if table is None:
            table = pd.read_csv(f'data/l2t/all_csv/{table_name}', delimiter="#")
        args = []
        for a in logic_json['args']:
            if isinstance(a, dict):
                args.append(ProgramTree.execute(table_name, a, table))
            else:
                if a == 'all_rows':
                    args.append(table)
                else:
                    args.append(a)

        return APIs[logic_json['func']]['function'](*args)

    @staticmethod
    def logic_json_to_str(logic_json):
        func = logic_json['func']
        args = [ProgramTree.logic_json_to_str(arg) if isinstance(arg, dict) else arg for arg in logic_json['args']]
        return APIs[func]['tostr'](*args)

    def __repr__(self):
        return json.dumps(self.logic_json, indent=2)

    def __str__(self):
        return json.dumps(self.logic_json)


class ProgramTreeBatch:
    def __init__(self, program_trees, vocab, tokenizer=None, cuda=False):
        # take some training files as input
        # This class needs to provide:
        # give the parent-action of another action
        # give the field-type of the action indexed by dfs order. The field type will come from the parent
        self.program_trees = program_trees
        self.vocab = vocab
        self.cuda = cuda
        self.max_num_actions = max(len(p.action_list) for p in self.program_trees)
        self.sent_list = [p.sent for p in self.program_trees if p.sent is not None]
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
        return len(self.program_trees)

    def get_parent_action_ids(self, curr_idx):
        ids = []
        for pt in self.program_trees:
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
        for pt in self.program_trees:
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
            copy_tok = False
            token_pos_list = []
            for tok_idx, tok in enumerate(self.input_ids[pt_id]):
                tok = get_val(tok)
                if tok == self.tokenizer_dict[start_tok]:
                    inside1 = True
                    continue
                elif tok == self.tokenizer_dict['fval_end_tok'] and inside1:
                    copy_tok = True
                    continue
                elif tok == self.tokenizer_dict[end_tok]:
                    inside1 = False
                    copy_tok = False
                    continue
                if copy_tok:
                    token_pos_list.append(tok_idx)
                    copy_tok = False
            return token_pos_list

        def extract2(pt_id, tok_list):
            i, j = 0, 0
            token_pos_list = []
            buff_list = []
            while i < len(self.input_ids[pt_id]):
                curr_tok = self.input_ids[pt_id][i]
                curr_tok = get_val(curr_tok)
                if curr_tok == tok_list[j]:
                    buff_list.append(i)
                    i += 1
                    j += 1
                    if len(buff_list) == len(tok_list):
                        token_pos_list.append(i)
                        buff_list = []
                        j = 0
                else:
                    if len(buff_list) == 0:
                        i += 1
                    buff_list = []
                    j = 0
            return token_pos_list

        for curr_ac_ix in range(self.max_num_actions):
            func_idx_row = []
            func_mask_row = []
            copy_mask_row = []
            # of size (batch,)
            parent_field_ids = self.get_parent_field_ids(curr_ac_ix)

            for pt_id, pt in enumerate(self.program_trees):
                func_id = func_mask = copy_mask = 0
                if curr_ac_ix < len(pt.action_list):
                    action = pt.action_list[curr_ac_ix][1]
                    action_info = pt.action_list[curr_ac_ix]

                    if action_info[0] == 'func':
                        func_id = self.vocab['actions'][action]
                        func_mask = 1
                        # avoid nan in softmax
                        self.generic_copy_mask[curr_ac_ix, pt_id, 0] = 1.
                    else:
                        # It's a copy token
                        field_type = self._get_field_from_id(get_val(parent_field_ids[pt_id]))
                        if field_type == 'n':
                            tok_pos_list = extract1(pt_id, 'n_start_tok', 'n_end_tok')
                        elif 'header' in field_type:
                            tok_pos_list = extract1(pt_id, 'hdr_start_tok', 'hdr_end_tok')
                        elif 'memory' in field_type:
                            tok_pos_list = extract1(pt_id, 'ent_start_tok', 'ent_end_tok')
                        else:
                            raise ValueError(f"Action {action_info} doesn't match field type {field_type}")
                        self.generic_copy_mask[curr_ac_ix, pt_id, tok_pos_list] = 1.
                        # avoid nan in softmax trick
                        if len(tok_pos_list) == 0:
                            self.generic_copy_mask[curr_ac_ix, pt_id, 0] = 1.

                        copy_utterance = '^# ' + action_info[1] + ' #^'
                        copy_utterance_toks = tokenizer.encode(copy_utterance, add_special_tokens=False)
                        tok_pos_list = extract2(pt_id, copy_utterance_toks)
                        self.copy_token_idx_mask[curr_ac_ix, pt_id, tok_pos_list] = 1.
                        copy_mask = 1
                else:
                    # avoid nan
                    self.generic_copy_mask[curr_ac_ix, pt_id, 0] = 1.

                func_idx_row.append(func_id)
                func_mask_row.append(func_mask)
                copy_mask_row.append(copy_mask)

            self.func_idx_matrix.append(func_idx_row)
            self.func_mask.append(func_mask_row)
            self.copy_mask.append(copy_mask_row)

        T = torch.cuda if self.cuda else torch
        self.func_idx_matrix = T.LongTensor(self.func_idx_matrix)
        self.func_mask = T.FloatTensor(self.func_mask).bool()
        self.copy_mask = T.FloatTensor(self.copy_mask).bool()
        self.copy_token_idx_mask = torch.from_numpy(self.copy_token_idx_mask).bool()
        self.generic_copy_mask = torch.from_numpy(self.generic_copy_mask).bool()
        if self.cuda:
            self.copy_token_idx_mask = self.copy_token_idx_mask.cuda()
            self.generic_copy_mask = self.generic_copy_mask.cuda()
        return


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, encoder_hidden_size):
        super(PointerNet, self).__init__()
        self.proj_linear = nn.Linear(encoder_hidden_size, query_vec_size, bias=False)

    def forward(self, sent_encodings, copy_token_mask, query_vec, get_log_prob=False):
        """
        :param sent_encodings: (batch_size, seq_len, encoder_hidden_size)
        :param copy_token_mask: (action_seq_len, batch_size, seq_len) contains bool which indicates the positions to potentially copy
        :param query_vec: (action_seq_len, batch_size, attn_vector_size)
        :param get_log_prob: Set True to get log probs instead of probs
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
        weights.data.masked_fill_(~copy_token_mask, -float('inf'))

        # (action_seq_len, batch_size, seq_len)
        ptr_weights = F.softmax(weights, dim=-1) if not get_log_prob else F.log_softmax(weights, dim=-1)
        return ptr_weights


class BeamSearchNode:
    """
    Represents a (partial) program in the decoding phase
    """

    def __init__(self):
        self.finished = False
        self.finished_action_idx = None
        # Shape - (n_actions, hidden_vector_size)
        self.hidden_vectors = []
        # Shape - (n_actions, ). Contains ('func', func_id) or ('tok', gen_tok)
        self.history_actions = []
        # Shape - (n_actions, ). Contains ptrs to parents for prev actions. Store -1 in case of no parents
        self.all_parent_ptrs = [None]
        # Stores the parent pointer for the frontier node
        self.curr_parent_ptr = None
        self.curr_field = None
        self.score = 0

    def clone(self, log_prob, act):
        tmp = copy.deepcopy(self)
        tmp.update_score(log_prob)
        tmp.history_actions.append(act)
        return tmp

    def update_score(self, log_prob):
        self.score += log_prob
        return

    def __hash__(self):
        return hash((tuple(self.history_actions), self.score))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.history_actions, self.score) == (other.history_actions, other.score)


class ProgramLSTM(nn.Module):
    """
    Takes as input a linked sentence and constructs a logic form AST recursively (left-to-right dfs).
    """

    def __init__(self, action_embed_size, field_embed_size, decoder_hidden_size,
                 attn_vec_size, dropout, device_str='cpu', model_name='roberta-base'):
        super(ProgramLSTM, self).__init__()
        self.action_embed_size = action_embed_size
        self.attn_vec_size = attn_vec_size
        self.field_embed_size = field_embed_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_name = model_name
        self.device = torch.device(device_str)
        with open('data/logic_form_vocab.json') as fp:
            self.vocab = json.load(fp)
        self.inv_vocab = {'actions': {v: k for k, v in self.vocab['actions'].items()},
                          'fields': {v: k for k, v in self.vocab['fields'].items()}}

        if 'gpt2' in model_name:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif 'roberta' in model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.tokenizer.add_tokens(NEW_TOKENS)
        self.tokenizer_dict = {
            'ent_start_tok': self.tokenizer.convert_tokens_to_ids('[ENT_START]'),
            'ent_end_tok': self.tokenizer.convert_tokens_to_ids('[ENT_END]'),
            'hdr_start_tok': self.tokenizer.convert_tokens_to_ids('[HDR_START]'),
            'hdr_end_tok': self.tokenizer.convert_tokens_to_ids('[HDR_END]'),
            'n_start_tok': self.tokenizer.convert_tokens_to_ids('[N_START]'),
            'n_end_tok': self.tokenizer.convert_tokens_to_ids('[N_END]'),
            'fval_start_tok': self.tokenizer.convert_tokens_to_ids('^#'),
            'fval_end_tok': self.tokenizer.convert_tokens_to_ids('#^')
        }

        if 'gpt2' in model_name:
            self.encoder = GPT2Model.from_pretrained(model_name)
            encoder_hidden_size = self.encoder.config.n_embd
        elif 'roberta' in model_name:
            self.encoder = RobertaModel.from_pretrained(model_name)
            encoder_hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = BertModel.from_pretrained(model_name)
            encoder_hidden_size = self.encoder.config.hidden_size
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        # encoder_hidden_size = self.encoder.config.hidden_size
        self.encoder.to(self.device)

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

        if device_str != 'cpu':
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
        :param mask: (batch_size, seq_len) indicating which values are not pad tokens
        """
        # attn_weight - (batch_size, seq_len)
        # bmm multiplies two tensors: b,m,p with b,p,n to get b,m,n
        attn_weight = torch.bmm(transformed_sent_encodings, h_t.unsqueeze(2)).squeeze(2)
        if mask is not None:
            attn_weight.data.masked_fill_(~mask, -float('inf'))
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
        if self.device != torch.device('cpu'):
            # (batch_size, seq_len)
            input_ids = padded_sequences['input_ids'].cuda()
            mask = padded_sequences['attention_mask'].cuda()
        else:
            input_ids = padded_sequences['input_ids']
            mask = padded_sequences['attention_mask']
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=mask)
        # (batch_size, seq_len, encoder_hidden_size)
        hidden_states = encoder_output.last_hidden_state
        return hidden_states, mask.bool()

    def init_decoder_state(self, enc_hidden_states):
        """
        Compute the initial decoder hidden state and cell state from the last state of encoder

        Args:
            enc_hidden_states: of shape (batch_size, seq_len, encoder_hidden_size)

        Returns: (batch_size, decoder_hidden_size) h_0, (batch_size, decoder_hidden_size) c_0

        """
        sel_idx = -1 if 'gpt2' in self.model_name else 0
        enc_last_state = enc_hidden_states[:, sel_idx, :]
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
                            prev_action_embed = zero_action_embed
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
        batch = ProgramTreeBatch(program_trees, self.vocab, self.tokenizer, cuda=self.device != torch.device('cpu'))
        sent_encodings, pad_masks = self.encode(batch.padded_sequences)
        dec_init_vec = self.init_decoder_state(sent_encodings)
        # query_vectors are attention hidden states h_t~ of the decoder
        # shape - (action_seq_len, batch_size, attn_vector_size)
        query_vectors = self.decode(batch, sent_encodings, dec_init_vec, pad_masks)

        # shape - (action_seq_len, batch_size, num_actions)
        apply_func_prob = F.softmax(self.action_readout(query_vectors), dim=-1)

        # shape - (action_seq_len, batch_size)
        target_apply_func_prob = torch.gather(apply_func_prob, dim=2,
                                              index=batch.func_idx_matrix.unsqueeze(2)).squeeze(2)

        # pointer network copying scores over source tokens
        # (action_seq_len, batch_size, seq_len)
        copy_prob = self.pointer_net(sent_encodings, batch.generic_copy_mask, query_vectors)

        # marginalize over the copy probabilities that generate the correct token
        # (action_seq_len, batch_size)
        target_copy_prob = torch.sum(copy_prob * batch.copy_token_idx_mask, dim=-1)

        # (action_seq_len, batch_size)
        action_prob = target_apply_func_prob * batch.func_mask + target_copy_prob * batch.copy_mask
        action_mask_pad = torch.eq(batch.func_mask + batch.copy_mask, 0.)
        action_mask = 1. - action_mask_pad.float()
        action_prob.data.masked_fill_(action_mask_pad.bool(), 1.e-7)
        if torch.any(action_prob == 0):
            # print("Action prob 0")
            action_prob.data.masked_fill_(action_prob == 0, 1.e-7)
        # print('uiop', torch.any(action_prob == 0))
        action_prob = action_prob.log() * action_mask
        scores = torch.sum(action_prob, dim=0)
        return scores

    def parse(self, padded_sequences, max_actions, train_rl=False):
        """Do a greedy search to generate target actions given an input sentence

        Args:
            padded_sequences: dict containing input_ids & attention mask of shape (batch_size, max_seq_len/seq_len)
            max_actions: Integer indicating the maximum actions to generate
            train_rl: If parsing is used while training with an RL objective

        Returns:
        """
        if self.device != torch.device('cpu'):
            # (batch_size, seq_len)
            input_ids = padded_sequences['input_ids'].cuda()
            T = torch.cuda
        else:
            input_ids = padded_sequences['input_ids']
            T = torch

        batch_size = input_ids.size(0)
        finished = [False for _ in range(batch_size)]
        finished_action_idx = [None for _ in range(batch_size)]
        log_probs, values = [], []
        if not train_rl:
            self.eval()

        sent_encodings, pad_masks = self.encode(padded_sequences)
        hc_pair = self.init_decoder_state(sent_encodings)
        transformed_sent_encodings = self.attn_1_linear(sent_encodings)
        zero_action_embed = self.new_tensor(self.action_embed_size).zero_()

        history_states = []
        # n_actions indicates actions taken before (history)
        # Shape - (n_actions, batch_size). Contains ('func', func_id) or ('tok', gen_tok)
        history_actions = []
        # Shape - (n_actions, batch_size). Contains ptrs to parents for prev actions. Store -1 in case of no parents
        all_parent_ptrs = [[None] * batch_size]
        parent_ptrs = []

        for t in range(max_actions):
            if all(finished):
                break
            if t == 0:
                x = self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_()
            else:
                prev_action_embeds = []
                # shape - (batch_size, )
                for batch_id in range(batch_size):
                    if not finished[batch_id]:
                        prev_action = history_actions[t - 1][batch_id]
                        if prev_action[0] == 'func':
                            prev_action_embed = self.action_embed.weight[prev_action[1]]
                        else:
                            prev_action_embed = zero_action_embed
                    else:
                        prev_action_embed = zero_action_embed

                    prev_action_embeds.append(prev_action_embed)
                prev_action_embeds = torch.stack(prev_action_embeds)

                # (batch_size,) Contains the ids of parent_actions
                parent_actions_ids = [history_actions[parent_ptrs[batch_id]][batch_id][1]
                                      if not finished[batch_id] else 0
                                      for batch_id in range(batch_size)]
                parent_actions_ids = T.LongTensor(parent_actions_ids)
                # (batch_size, action_embed_size)
                parent_action_embed = self.action_embed(parent_actions_ids)
                # (batch_size, ) stores curr field names
                curr_fields = []
                parent_field_ids = []
                for batch_id in range(batch_size):
                    if not finished[batch_id]:
                        parent_action_idx = parent_ptrs[batch_id]
                        # ('func', func_id)
                        p_action_info = history_actions[parent_action_idx][batch_id]
                        p_action_name = self.inv_vocab['actions'][p_action_info[1]]
                        arg_idx = len([a for a in all_parent_ptrs if a[batch_id] == parent_action_idx]) - 1
                        ftype = APIs[p_action_name]['model_args'][arg_idx]
                        parent_field_ids.append(self.vocab['fields'][ftype])
                        curr_fields.append(ftype)
                    else:
                        parent_field_ids.append(0)
                        curr_fields.append('no_field')

                parent_field_ids = T.LongTensor(parent_field_ids)
                parent_field_embed = self.field_embed(parent_field_ids)
                inputs = [prev_action_embeds, prev_attn_t, parent_action_embed, parent_field_embed]

                parent_states = torch.stack([
                    history_states[parent_ptrs[batch_id] if not finished[batch_id] else 0][0][batch_id]
                    for batch_id in range(batch_size)])

                inputs.append(parent_states)
                x = torch.cat(inputs, dim=-1)

            # (batch_size, decoder_hidden_size) - ht, ct
            # (batch_size, attn_vector_size)
            hc_pair, attn_t = self.decoder_step(
                x, hc_pair, sent_encodings, transformed_sent_encodings, attn_token_mask=pad_masks)

            # Choose between the following two based on the field embed (is it a row, is it a hdr, mem, n?)
            # shape - (batch_size, num_actions)
            hist_act_row = []
            for batch_id in range(batch_size):
                if finished[batch_id]:
                    hist_act_row.append(('func', 0))
                    continue
                if t == 0 or curr_fields[batch_id] in ['row', 'obj']:
                    # shape - (num_actions,)
                    apply_func_prob = F.softmax(self.action_readout(attn_t[batch_id]), dim=-1)
                    if not train_rl:
                        # shape - (1,)
                        next_func = torch.argmax(apply_func_prob, dim=-1).unsqueeze(-1)
                    else:
                        action_probs = torch.distributions.categorical.Categorical(apply_func_prob)
                        next_func = action_probs.sample()
                        log_prob = action_probs.log_prob(next_func)
                        log_probs.append(log_prob)
                        values.append(self.value_linear(attn_t[batch_id]))
                    hist_act_row.append(('func', next_func.item()))
                else:
                    # create copy_mask which has 1 where we can copy entities
                    field_type = curr_fields[batch_id]
                    if field_type == 'n':
                        start_tok, end_tok = 'n_start_tok', 'n_end_tok'
                    elif 'header' in field_type:
                        start_tok, end_tok = 'hdr_start_tok', 'hdr_end_tok'
                    elif 'memory' in field_type:
                        start_tok, end_tok = 'ent_start_tok', 'ent_end_tok'
                    else:
                        raise ValueError(f"Wrong branch! Can't copy for the field {field_type}")

                    inside = False
                    copy_tok = False
                    token_pos_list = []

                    for tok_idx, tok in enumerate(input_ids[batch_id]):
                        tok = get_val(tok)
                        if tok == self.tokenizer_dict[start_tok]:
                            inside = True
                            continue
                        elif tok == self.tokenizer_dict['fval_end_tok'] and inside:
                            copy_tok = True
                            continue
                        elif tok == self.tokenizer_dict[end_tok]:
                            inside = False
                            copy_tok = False
                            continue
                        if copy_tok:
                            token_pos_list.append(tok_idx)
                            copy_tok = False
                    # (seq_len, )
                    copy_mask = np.zeros((input_ids.size(1),), dtype='float32')
                    copy_mask[token_pos_list] = 1
                    copy_mask = torch.from_numpy(copy_mask).bool()
                    if not torch.any(copy_mask) and train_rl:
                        return None, log_probs, values
                    if self.device != torch.device('cpu'):
                        copy_mask = copy_mask.cuda()
                    # (1, 1, seq_len)
                    copy_mask = copy_mask.unsqueeze(0).unsqueeze(0)
                    copy_prob = self.pointer_net(
                        sent_encodings[batch_id].unsqueeze(0), copy_mask,
                        attn_t[batch_id, :].unsqueeze(0).unsqueeze(0))
                    copy_prob = copy_prob.squeeze(0).squeeze(0)
                    if not train_rl:
                        next_token_idx = torch.argmax(copy_prob, dim=-1).unsqueeze(-1)
                    else:
                        action_probs = torch.distributions.categorical.Categorical(copy_prob)
                        next_token_idx = action_probs.sample()
                        log_prob = action_probs.log_prob(next_token_idx)
                        log_probs.append(log_prob)
                        values.append(self.value_linear(attn_t[batch_id]))
                    hist_act_row.append(('tok', self.tokenizer.decode(input_ids[batch_id][next_token_idx])))

            history_actions.append(hist_act_row)

            # check if any outstanding actions are left
            parent_ptrs = [None] * batch_size
            for batch_id in range(batch_size):
                if finished[batch_id]:
                    continue
                all_actions = [act[batch_id] for act in history_actions]
                pseudo_funcs = [self.vocab['actions']['nop'], self.vocab['actions']['all_rows']]
                func2sat = {a_idx: False for a_idx, a in enumerate(all_actions)
                            if a[0] == 'func' and a[1] not in pseudo_funcs}
                func2numargs = {a_idx: 0 for a_idx in func2sat.keys()}
                for a_idx, a in enumerate(all_actions):
                    parent_action_idx = all_parent_ptrs[a_idx][batch_id]
                    if parent_action_idx is not None:
                        func2numargs[parent_action_idx] += 1
                for a_idx, num_args in func2numargs.items():
                    fn = self.inv_vocab['actions'][all_actions[a_idx][1]]
                    if len(APIs[fn]['model_args']) <= num_args:
                        func2sat[a_idx] = True
                finished[batch_id] = all(func2sat.values())
                if finished[batch_id]:
                    finished_action_idx[batch_id] = t
                if not finished[batch_id]:
                    parent_ptrs[batch_id] = max(a_idx for a_idx in func2sat.keys() if not func2sat[a_idx])

            all_parent_ptrs.append(parent_ptrs)
            history_states.append(hc_pair)
            prev_attn_t = attn_t

        # (batch_size, n_actions_variable)
        new_history_actions = [[] for _ in range(batch_size)]
        for batch_id in range(batch_size):
            tot_actions = (finished_action_idx[batch_id] + 1 if finished_action_idx[batch_id] is not None
                           else max_actions)
            for action_idx in range(tot_actions):
                atype, aid = history_actions[action_idx][batch_id]
                if atype == 'func':
                    new_history_actions[batch_id].append(('func', self.inv_vocab['actions'][aid]))
                elif atype == 'tok':
                    new_history_actions[batch_id].append(('tok', aid))

        if not train_rl:
            return new_history_actions
        else:
            return new_history_actions, log_probs, values

    def beam_parse(self, padded_sequences, max_actions, top_k=1, beam_width=1):
        """Do a beam search to generate target actions given an input sentence

        Args:
            padded_sequences: dict containing input_ids & attention mask of shape (batch_size, max_seq_len/seq_len)
            max_actions: Integer indicating the maximum actions to generate
            top_k: Top k results to return for each input seq
            beam_width: Beam width for beam search

        Returns:
        """
        assert beam_width >= top_k

        if self.device != torch.device('cpu'):
            # (batch_size, seq_len)
            input_ids = padded_sequences['input_ids'].cuda()
            T = torch.cuda
        else:
            input_ids = padded_sequences['input_ids']
            T = torch

        batch_size = input_ids.size(0)
        self.eval()

        sent_encodings, pad_masks = self.encode(padded_sequences)
        if beam_width > 1:
            sent_encodings = tile(sent_encodings, 0, beam_width, self.new_long_tensor)
            pad_masks = tile(pad_masks, 0, beam_width, self.new_long_tensor)
        hc_pair = self.init_decoder_state(sent_encodings)
        transformed_sent_encodings = self.attn_1_linear(sent_encodings)
        zero_action_embed = self.new_tensor(self.action_embed_size).zero_()

        # Shape - (batch_size, beam_width)
        beam_search_nodes = [[BeamSearchNode() for _ in range(beam_width)] for _ in range(batch_size)]

        for t in range(max_actions):
            if all(bsn.finished for bsn_arr in beam_search_nodes for bsn in bsn_arr):
                break
            if t == 0:
                x = self.new_tensor(batch_size * beam_width, self.decoder_lstm.input_size).zero_()
            else:
                prev_action_embeds = []
                # shape - (batch_size x beam_width, )
                for batch_id in range(batch_size):
                    for bw_id in range(beam_width):
                        bsn = beam_search_nodes[batch_id][bw_id]
                        if not bsn.finished:
                            prev_action = bsn.history_actions[t - 1]
                            if prev_action[0] == 'func':
                                prev_action_embed = self.action_embed.weight[prev_action[1]]
                            else:
                                prev_action_embed = zero_action_embed
                        else:
                            prev_action_embed = zero_action_embed
                        prev_action_embeds.append(prev_action_embed)
                prev_action_embeds = torch.stack(prev_action_embeds)

                # (batch_size x beam_width, ) Contains the ids of parent_actions
                parent_actions_ids = []
                for batch_id in range(batch_size):
                    for bw_id in range(beam_width):
                        bsn = beam_search_nodes[batch_id][bw_id]
                        if not bsn.finished:
                            parent_actions_ids.append(bsn.history_actions[bsn.curr_parent_ptr][1])
                        else:
                            parent_actions_ids.append(0)
                parent_actions_ids = T.LongTensor(parent_actions_ids)
                # (batch_size x beam_width, action_embed_size)
                parent_action_embed = self.action_embed(parent_actions_ids)
                # (batch_size x beam_width, ) stores curr field names
                parent_field_ids = []
                for batch_id in range(batch_size):
                    for bw_id in range(beam_width):
                        bsn = beam_search_nodes[batch_id][bw_id]
                        if not bsn.finished:
                            # ('func', func_id)
                            p_action_info = bsn.history_actions[bsn.curr_parent_ptr]
                            p_action_name = self.inv_vocab['actions'][p_action_info[1]]
                            arg_idx = len([a for a in bsn.all_parent_ptrs if a == bsn.curr_parent_ptr]) - 1
                            ftype = APIs[p_action_name]['model_args'][arg_idx]
                            parent_field_ids.append(self.vocab['fields'][ftype])
                            bsn.curr_field = ftype
                        else:
                            parent_field_ids.append(0)
                            bsn.curr_field = 'no_field'

                parent_field_ids = T.LongTensor(parent_field_ids)
                # (batch_size x beam_width, field_embed_size)
                parent_field_embed = self.field_embed(parent_field_ids)

                # (batch_size x beam_width, decoder_hidden_size)
                parent_states = []
                for batch_id in range(batch_size):
                    for bw_id in range(beam_width):
                        bsn = beam_search_nodes[batch_id][bw_id]
                        if not bsn.finished:
                            parent_states.append(bsn.hidden_vectors[bsn.curr_parent_ptr])
                        else:
                            parent_states.append(bsn.hidden_vectors[0])
                parent_states = torch.stack(parent_states)

                inputs = [prev_action_embeds, prev_attn_t, parent_action_embed,
                          parent_field_embed, parent_states]
                x = torch.cat(inputs, dim=-1)

            # (batch_size x beam_width, decoder_hidden_size) - ht, ct
            # (batch_size x beam_width, attn_vector_size)
            hc_pair, attn_t = self.decoder_step(
                x, hc_pair, sent_encodings, transformed_sent_encodings, attn_token_mask=pad_masks)

            unflat_attn_t = attn_t.view(batch_size, beam_width, self.attn_vec_size)
            unflat_hidd_vec = hc_pair[0].view(batch_size, beam_width, self.decoder_hidden_size)
            # Choose between the following two based on the field embed (is it a row, is it a hdr, mem, n?)
            # shape - (batch_size x beam_width, num_actions)
            new_beam_search_nodes = {}
            for batch_id in range(batch_size):
                for bw_id in range(beam_width):
                    bsn = beam_search_nodes[batch_id][bw_id]
                    new_beam_search_nodes[(batch_id, bw_id)] = []
                    if bsn.finished:
                        bsn.history_actions.append(('func', 0))
                        new_beam_search_nodes[(batch_id, bw_id)].append(bsn)
                        continue
                    if t == 0 or bsn.curr_field in ['row', 'obj']:
                        # shape - (num_actions,)
                        apply_func_log_prob = F.log_softmax(
                            self.action_readout(unflat_attn_t[batch_id][bw_id]), dim=-1)
                        for func_id, log_prob in enumerate(apply_func_log_prob.tolist()):
                            if not np.isfinite(log_prob):
                                continue
                            new_beam_search_nodes[(batch_id, bw_id)].append(bsn.clone(log_prob, ('func', func_id)))
                    else:
                        # create copy_mask which has 1 where we can copy entities
                        if bsn.curr_field == 'n':
                            start_tok, end_tok = 'n_start_tok', 'n_end_tok'
                        elif 'header' in bsn.curr_field:
                            start_tok, end_tok = 'hdr_start_tok', 'hdr_end_tok'
                        elif 'memory' in bsn.curr_field:
                            start_tok, end_tok = 'ent_start_tok', 'ent_end_tok'
                        else:
                            raise ValueError(f"Wrong branch! Can't copy for the field {bsn.curr_field}")

                        inside = False
                        copy_tok = False
                        token_pos_list = []

                        for tok_idx, tok in enumerate(input_ids[batch_id]):
                            tok = get_val(tok)
                            if tok == self.tokenizer_dict[start_tok]:
                                inside = True
                                continue
                            elif tok == self.tokenizer_dict['fval_end_tok'] and inside:
                                copy_tok = True
                                continue
                            elif tok == self.tokenizer_dict[end_tok]:
                                inside = False
                                copy_tok = False
                                continue
                            if copy_tok:
                                token_pos_list.append(tok_idx)
                                copy_tok = False
                        # (seq_len, )
                        copy_mask = np.zeros((input_ids.size(1),), dtype='float32')
                        copy_mask[token_pos_list] = 1
                        copy_mask = torch.from_numpy(copy_mask).bool()
                        if not torch.any(copy_mask):
                            copy_mask[0] = True
                        if self.device != torch.device('cpu'):
                            copy_mask = copy_mask.cuda()
                        # (1, 1, seq_len)
                        copy_mask = copy_mask.unsqueeze(0).unsqueeze(0)
                        copy_log_prob = self.pointer_net(
                            sent_encodings[batch_id].unsqueeze(0), copy_mask,
                            attn_t[batch_id, :].unsqueeze(0).unsqueeze(0), get_log_prob=True)
                        copy_log_prob = copy_log_prob.squeeze(0).squeeze(0)
                        for tok_idx, log_prob in enumerate(copy_log_prob.tolist()):
                            if not np.isfinite(log_prob):
                                continue
                            next_tok = self.tokenizer.decode(input_ids[batch_id][tok_idx])
                            new_beam_search_nodes[(batch_id, bw_id)].append(bsn.clone(log_prob, ('tok', next_tok)))

            # check if any outstanding actions are left
            pseudo_funcs = [self.vocab['actions']['nop'], self.vocab['actions']['all_rows']]
            for batch_id in range(batch_size):
                for bw_id in range(beam_width):
                    for bsn in new_beam_search_nodes[(batch_id, bw_id)]:
                        if bsn.finished:
                            continue
                        func2sat = {a_idx: False for a_idx, a in enumerate(bsn.history_actions)
                                    if a[0] == 'func' and a[1] not in pseudo_funcs}
                        func2numargs = {a_idx: 0 for a_idx in func2sat.keys()}
                        for a_idx, a in enumerate(bsn.history_actions):
                            parent_action_idx = bsn.all_parent_ptrs[a_idx]
                            if parent_action_idx is not None:
                                func2numargs[parent_action_idx] += 1
                        for a_idx, num_args in func2numargs.items():
                            fn = self.inv_vocab['actions'][bsn.history_actions[a_idx][1]]
                            if len(APIs[fn]['model_args']) <= num_args:
                                func2sat[a_idx] = True
                        bsn.finished = all(func2sat.values())
                        if bsn.finished:
                            bsn.finished_action_idx = t
                        if not bsn.finished:
                            bsn.curr_parent_ptr = max(a_idx for a_idx in func2sat.keys() if not func2sat[a_idx])
                        bsn.all_parent_ptrs.append(bsn.curr_parent_ptr)
                        bsn.hidden_vectors.append(unflat_hidd_vec[batch_id][bw_id])

            prev_attn_t = attn_t
            beam_search_nodes = [[] for _ in range(batch_size)]
            for batch_id in range(batch_size):
                candidate_bsns = [new_beam_search_nodes[(batch_id, bw_id)] for bw_id in range(beam_width)]
                candidate_bsns = [x for y in candidate_bsns for x in y]
                candidate_bsns = list(set(candidate_bsns))
                candidate_bsns = sorted(candidate_bsns, key=lambda bsn: bsn.score, reverse=True)
                while len(candidate_bsns) < beam_width:
                    candidate_bsns.append(copy.deepcopy(candidate_bsns[0]))
                beam_search_nodes[batch_id] = candidate_bsns[:beam_width]

        # (batch_size, top_k, n_actions_variable)
        new_history_actions = [[[] for _ in range(top_k)] for _ in range(batch_size)]
        for batch_id in range(batch_size):
            selected_bsns = sorted(beam_search_nodes[batch_id], key=lambda bsn: bsn.score, reverse=True)
            selected_bsns = selected_bsns[:top_k]
            for k_id, bsn in enumerate(selected_bsns):
                tot_actions = bsn.finished_action_idx + 1 if bsn.finished_action_idx is not None else max_actions
                for action_idx in range(tot_actions):
                    atype, aid = bsn.history_actions[action_idx]
                    if atype == 'func':
                        new_history_actions[batch_id][k_id].append(('func', self.inv_vocab['actions'][aid]))
                    elif atype == 'tok':
                        new_history_actions[batch_id][k_id].append(('tok', aid))

        return new_history_actions

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            'args': (self.action_embed_size, self.field_embed_size,
                     self.decoder_hidden_size, self.attn_vec_size,
                     self.dropout.p, self.model_name),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
        return

    @classmethod
    def load(cls, model_path, cuda=False):
        device_str = 'cuda' if cuda else 'cpu'
        params = torch.load(model_path, map_location=torch.device(device_str))
        (action_embed_size, field_embed_size, decoder_hidden_size,
         attn_vec_size, dropout, gpt_model) = params['args']

        plstm = cls(action_embed_size, field_embed_size, decoder_hidden_size,
                    attn_vec_size, dropout, device_str, gpt_model)
        saved_state = params['state_dict']
        if 'rl' in model_path:
            plstm.value_linear = nn.Linear(plstm.attn_vec_size, 1, bias=False)
        plstm.load_state_dict(saved_state)
        if cuda:
            plstm = plstm.cuda()
        plstm.eval()
        return plstm

    @staticmethod
    def train_program_lstm(args):
        args.dbg_sent, args.train_set = '', False
        print(args)
        save_path = args.save_base_dir / Path('plstm_models/')
        save_path.mkdir(exist_ok=True, parents=True)
        device_str = 'cuda' if args.cuda else 'cpu'
        device = torch.device(device_str)
        if args.resume_train:
            model = ProgramLSTM.load(args.model_path, args.cuda)
        else:
            model = ProgramLSTM(32, 32, 256, 256, 0.2, device_str, args.model)
        model.to(device)

        with open('data/programs_filtered.json') as fp:
            data = json.load(fp)

        all_programs = []
        for entry in data:
            for prog in entry[-1]:
                pt = ProgramTree.from_str(prog)
                pt.sent = entry[3]
                all_programs.append(pt)

        recording_time = datetime.now().strftime('%m_%d_%H_%M')
        optimizer = AdamW(model.parameters(), args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps,
            num_training_steps=args.epochs * (len(all_programs) // args.batch_size + 1))

        global_step = avg_loss = start_epoch = 0

        if args.resume_train:
            checkpoint = torch.load(args.ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            recording_time = checkpoint['recording_time']
            global_step = checkpoint['total_steps']
            start_epoch = checkpoint['epochs_finished'] + 1
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        tb_writer = SummaryWriter(log_dir=f'tensorboard/p-lstm/{recording_time}')
        print("Recording Time:", recording_time)

        for epoch_idx in range(start_epoch, args.epochs):
            print("start training {}th epoch".format(epoch_idx))
            model.train()
            random.shuffle(all_programs)
            for idx in tqdm(range(0, len(all_programs), args.batch_size),
                            total=len(all_programs) // args.batch_size + 1):
                global_step += 1
                batch_progs = all_programs[idx:idx + args.batch_size]
                optimizer.zero_grad()
                ret_val = model.score(batch_progs)
                loss = -ret_val[0]
                avg_loss += torch.sum(loss).data.item()
                loss = torch.mean(loss)
                # with autograd.detect_anomaly():
                #     loss.backward()
                #     optimizer.step()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                scheduler.step()

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("Avg loss", avg_loss / args.every, global_step)
                    avg_loss = 0

            if (epoch_idx + 1) % args.save_every == 0:
                torch.save({
                    'recording_time': recording_time,
                    'total_steps': global_step,
                    'epochs_finished': epoch_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                    save_path / f'ws_ckpt_{epoch_idx:03}.pt')
                model.save(save_path / f'ws_model_{epoch_idx:03}.pt')
                model.eval()
                with torch.no_grad():
                    val_results = ProgramLSTM.get_model_results('data/plstm_valid.json', model, args)
                val_results = [(x[-1], x[-2] is not None) for x in val_results]
                ctr = Counter(val_results)
                ctr2 = {
                    'FN': ctr[(False, False)],
                    'FP': ctr[(False, True)],
                    'TP': ctr[(True, True)],
                }
                coverage = ctr2['TP'] / len(val_results)
                tb_writer.add_scalar("Validation coverage", coverage, epoch_idx)
                tb_writer.add_scalar("Validation FN", ctr2['FN'], epoch_idx)
                tb_writer.add_scalar("Validation FP", ctr2['FP'], epoch_idx)
                tb_writer.add_scalar("Validation TP", ctr2['TP'], epoch_idx)

        tb_writer.flush()
        tb_writer.close()
        return

    @staticmethod
    def train_rl_program_lstm(args):
        print(args)
        save_path = args.save_base_dir / Path('plstm_models/')
        save_path.mkdir(exist_ok=True, parents=True)
        device_str = 'cuda' if args.cuda else 'cpu'
        device = torch.device(device_str)
        # pick warm-start model
        model = ProgramLSTM.load(args.model_path, args.cuda)
        model.value_linear = nn.Linear(model.attn_vec_size, 1, bias=False)
        model.to(device)
        model.train()

        with open('data/programs_filtered.json') as fp:
            data = json.load(fp)

        train_data = []
        for entry in data:
            mask_val = (entry[4][0], str(entry[4][1]))
            train_data.append((entry[3], entry[0], mask_val))

        all_programs = []
        for entry in data:
            for prog in entry[-1]:
                pt = ProgramTree.from_str(prog)
                pt.sent = entry[3]
                all_programs.append(pt)

        recording_time = datetime.now().strftime('%m_%d_%H_%M')
        optimizer = optim.Adam(model.parameters(), args.lr)

        episode_idx = n_episode_idx = 0

        if args.resume_train:
            checkpoint = torch.load(args.ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            recording_time = checkpoint['recording_time']
            n_episode_idx = checkpoint['curr_episode']

        tb_writer = SummaryWriter(log_dir=f'tensorboard/p-lstm/{recording_time}')
        score_history = []
        random.seed(32)
        while True:
            random.shuffle(train_data)
            for trans_sent, table_name, mask_val in tqdm(train_data):
                episode_idx += 1
                if episode_idx <= n_episode_idx:
                    continue
                # print(f"Starting episode {episode_idx}")

                if torch.rand(1).item() * (1 - episode_idx / args.episodes) > 0.5:
                    batch_progs = np.random.choice(all_programs, args.batch_size, replace=False)
                    optimizer.zero_grad()
                    ret_val = model.score(batch_progs)
                    loss = -ret_val[0]
                    loss = torch.mean(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()

                padded_sequences = model.tokenizer(trans_sent, padding=True, truncation=True, return_tensors="pt")
                model_out, log_probs, values = model.parse(padded_sequences, args.max_actions, True)
                if model_out is None:
                    # reward -1 when no copy tokens can be found
                    act_list = None
                else:
                    act_list = model_out[0]

                ret_val = None
                try:
                    logic_json = ProgramTree.get_logic_json_from_action_list(act_list, trans_sent)
                    ret_val = ProgramTree.execute(table_name, logic_json)
                    is_accepted = check_if_accept(logic_json['func'], ret_val, mask_val[1])
                except Exception as err:
                    is_accepted = False

                rewards = [0] * len(log_probs)
                if is_accepted:
                    rewards[-1] = 1
                elif ret_val is not None:
                    rewards[-1] = 0
                else:
                    rewards[-1] = -1

                R = model.new_tensor([0])
                returns = [model.new_tensor([0])] * len(log_probs)
                advantage = model.new_tensor([0])
                advantages = [model.new_tensor([0])] * len(log_probs)
                for t in reversed(range(len(log_probs))):
                    # rewards, masks, actions, policies, values = steps[t]
                    reward, curr_value = rewards[t], values[t]
                    next_value = values[t + 1] if t + 1 < len(log_probs) else 0
                    R = reward + R
                    deltas = reward + next_value - curr_value
                    advantage = advantage + deltas
                    advantages[t] = advantage
                    returns[t] = R
                    # out[t] = actions, policies, values, returns, advantages
                advantages = torch.cat(advantages, dim=0)
                returns = torch.cat(returns, dim=0)
                log_probs = torch.stack(log_probs)
                values = torch.cat(values, dim=0)

                policy_loss = (-log_probs * advantages).sum()
                value_loss = (.5 * (values - returns) ** 2.).sum()

                loss = policy_loss + value_loss * args.value_coeff

                # policy_loss = []
                # returns = []
                # R = 0
                # for r in rewards[::-1]:
                #     R = r + 1 * R
                #     returns.insert(0, R)
                # returns = torch.FloatTensor(returns)
                # # returns = (returns - returns.mean()) / (returns.std(unbiased=False) + eps)
                # for log_prob, R in zip(log_probs, returns):
                #     policy_loss.append(-log_prob * R)
                # policy_loss = torch.stack(policy_loss).sum()
                # policy_loss.backward()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()

                tb_writer.add_scalar("ep_return", rewards[-1], episode_idx)
                score_history.append(rewards[-1])
                avg_score = np.mean(score_history[-args.avg_hist:])

                if episode_idx % args.avg_hist == 0:
                    tb_writer.add_scalar("avg_reward", avg_score, episode_idx)

                if episode_idx % args.save_every == 0:
                    torch.save({
                        'recording_time': recording_time,
                        'curr_episode': episode_idx,
                        'optimizer_state_dict': optimizer.state_dict()},
                        save_path / f'rl_ckpt_{episode_idx:06}.pt')
                    model.save(save_path / f'rl_model_{episode_idx:06}.pt')

                if episode_idx >= args.episodes:
                    break
            if episode_idx >= args.episodes:
                break

        tb_writer.flush()
        tb_writer.close()
        return

    @staticmethod
    def get_model_results(in_fname, model, args):
        assert args.top_k == 1 or len(args.ranker_model_path) > 0
        if len(args.ranker_model_path) > 0 and args.top_k > 1:
            from ranker_model import RobertaRanker
            ranker_model = RobertaRanker.load(args.ranker_model_path, args.cuda)
            device_str = 'cuda' if args.cuda else 'cpu'
            device = torch.device(device_str)
            ranker_model.to(device)
            ranker_model.eval()

        with open(in_fname) as fp:
            data = json.load(fp)

        results = []
        if len(args.dbg_sent) > 0:
            args.batch_size = 1
        for idx in tqdm(range(0, len(data), args.batch_size), total=len(data) // args.batch_size + 1):
            if len(args.dbg_sent) > 0:
                if data[idx][1] != args.dbg_sent:
                    continue
            entries = data[idx:idx + args.batch_size]

            if args.train_set:
                i1, i2 = 3, 4
            else:
                i1, i2 = -1, 3

            sent_list = [e[i1] for e in entries]
            table_names = [e[0] for e in entries]
            masked_vals = [e[i2] for e in entries]
            og_sent_list = [e[1] for e in entries]

            padded_sequences = model.tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                model_out = model.beam_parse(padded_sequences, args.max_actions, args.top_k, args.beam_width)

            for e_idx in range(len(entries)):
                trans_sent, table_name = sent_list[e_idx], table_names[e_idx]
                mask_val, og_sent = masked_vals[e_idx], og_sent_list[e_idx]
                mask_val = (mask_val[0], str(mask_val[1]))
                ranker_input = []
                prog_data = []
                sample_data = None
                for k_idx in range(args.top_k):
                    act_list = model_out[e_idx][k_idx]
                    logic_json, ret_val = None, None
                    try:
                        logic_json = ProgramTree.get_logic_json_from_action_list(act_list, trans_sent)
                        ret_val = ProgramTree.execute(table_name, logic_json)
                        is_accepted = check_if_accept(logic_json['func'], ret_val, mask_val[1])
                    except Exception as err:
                        is_accepted = False
                    try:
                        ljsonstr = ProgramTree.logic_json_to_str(logic_json)
                    except:
                        ljsonstr = None
                    if ljsonstr is not None:
                        ljsonstr += f'={ret_val}/{is_accepted}'
                        if args.top_k > 1:
                            label = ljsonstr[ljsonstr.rfind('/') + 1:] == 'True'
                            prog = ljsonstr[:ljsonstr.rfind('=')]
                            ip_sent = RobertaRanker._get_input_sent(trans_sent, prog, model)
                            ranker_input.append((ip_sent, label))
                            prog_data.append((act_list, ljsonstr, is_accepted))
                    sample_data = act_list, ljsonstr, is_accepted
                with torch.no_grad():
                    if len(ranker_input) > 0 and args.top_k > 1:
                        ranker_ip_sents, labels, = zip(*ranker_input)
                        labels = model.new_long_tensor(labels).unsqueeze(1)
                        padded_sequences = ranker_model.tokenizer(
                            ranker_ip_sents, padding=True, truncation=True, return_tensors="pt")
                        output = ranker_model(padded_sequences, labels)
                        probs = F.softmax(output.logits, dim=-1)
                        scores = probs[:, 1]
                        max_score_idx = torch.argmax(scores).item()
                        max_score = scores[max_score_idx].item()
                        act_list, ljsonstr, is_accepted = prog_data[max_score_idx]
                    else:
                        act_list, ljsonstr, is_accepted = sample_data
                        max_score = 1

                results.append((table_name, og_sent, trans_sent,
                                mask_val, act_list, max_score, ljsonstr, is_accepted))
        return results

    @staticmethod
    def test_program_lstm(args):
        print(args)
        save_path = Path('plstm_outputs/')
        save_path.mkdir(exist_ok=True)
        device_str = 'cuda' if args.cuda else 'cpu'
        device = torch.device(device_str)

        model = ProgramLSTM.load(args.model_path, args.cuda)
        model.to(device)
        model.eval()
        in_fname = 'data/programs_filtered.json' if args.train_set else 'data/plstm_test.json'
        results = ProgramLSTM.get_model_results(in_fname, model, args)
        with open(save_path / f'out_{args.out_id}.json', 'w') as fp:
            json.dump(results, fp, indent=2)
        return


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
    #
    # l_str = "greater_str_inv { all_rows ; venue ; belk gymnasium ; closed }=charlotte speedway/True"
    # a4 = ProgramTree.from_str(l_str)
    # print(repr(a4))
    # print(a4.action_list)
    #
    # with open('data/logic_form_vocab.json') as fp:
    #     vocab = json.load(fp)
    # # b = ProgramTreeDataset([a1, a2, a3, a4], vocab)
    # b = ProgramTreeBatch([a4], vocab)
    # print(b.get_parent_action_ids(5))
    # print(b.get_parent_field_ids(5))
    #
    # # ns = ProgramTree.transform_linked_sent(
    # #     '#sap g33k;5,2# be the team with the third highest #team number;0,3# in the #first championship;-1,-1# .',
    # #     pd.read_csv('data/l2t/all_csv/2-15584199-3.html.csv', delimiter="#")
    # # )
    # # print(ns)
    #
    # l_str = "hop { nth_argmax { all_rows ; total ; 2 } ; province }=chonburi/True"
    # a5 = ProgramTree.from_str(l_str)
    # print(repr(a5))
    # print(a5.action_list)
    #
    # with open('data/logic_form_vocab.json') as fp:
    #     vocab = json.load(fp)
    # b = ProgramTreeBatch([a5], vocab)
    # print(b.get_parent_action_ids(5))
    # print(b.get_parent_field_ids(5))

    # ns = ProgramTree.transform_linked_sent(
    #     '#chonburi;2,1# receive the 2nd highest #total;0,5# medal count in the #2008 thailand national game;-1,-1# .',
    #     pd.read_csv('data/l2t/all_csv/2-14892957-1.html.csv', delimiter="#")
    # )
    # print(ns)

    # l_str = "hop { nth_argmax { all_rows ; silver ; 2 } ; province }=chonburi/True"
    # a5 = ProgramTree.from_str(l_str,
    #                           '#chonburi;4,1# #province;0,1# win the second highest amount'
    #                           ' of #silver;0,3# medal in the #2009 thailand national game;-1,-1# .',
    #                           [
    #                               "rank",
    #                               "province",
    #                               "gold",
    #                               "silver",
    #                               "bronze",
    #                               "total"
    #                           ],
    #                           {
    #                               0: "num",
    #                               1: "str",
    #                               2: "num",
    #                               3: "num",
    #                               4: "num",
    #                               5: "num"
    #                           }
    #                           )
    # print(repr(a5))
    # print(a5.sent)
    # print(a5.action_list)

    # l_str = "hop { nth_argmin { all_rows ; length ; 2 } ; line }=sofia - dragoman/True"
    # a5 = ProgramTree.from_str(l_str,
    #                           'the second shortest #speed rail in europe;-1,-1# be the'
    #                           ' #sofia - dragoman;6,0# #line;0,0# .',
    #                           [
    #                               "line",
    #                               "speed",
    #                               "length",
    #                               "construction begun",
    #                               "expected start of revenue services"
    #                           ],
    #                           {
    #                               0: "str",
    #                               1: "num",
    #                               2: "num",
    #                               3: "num",
    #                               4: "num"
    #                           }
    #                           )
    # print(repr(a5))
    # print(a5.sent)
    # print(a5.action_list)

    l_str = ("greater_str_inv { all_rows ; artist ; dire straits ;"
             " release - year of first charted record }=barbra streisand/True")
    a5 = ProgramTree.from_str(l_str,
                              '#best - selling music artist;-1,-1# #dire straits;16,0# have '
                              'less #claimed sales;0,5# than #barbra streisand;8,0# .',
                              [
                                  "artist",
                                  "country of origin",
                                  "period active",
                                  "release - year of first charted record",
                                  "genre",
                                  "claimed sales"
                              ],
                              {
                                  0: "num",
                                  1: "str",
                                  2: "num",
                                  3: "num",
                                  4: "str",
                                  5: "num"
                              }
                              )
    print(repr(a5))
    print(a5.sent)
    print(a5.action_list)

    # with open('data/logic_form_vocab.json') as fp:
    #     vocab = json.load(fp)
    # b = ProgramTreeBatch([a5], vocab)
    # print(b.get_parent_action_ids(5))
    # print(b.get_parent_field_ids(5))
    return


def init_plstm_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', default=False, help='Weakly supervised training')
    parser.add_argument('--do_test', action='store_true', default=False, help='Test')
    parser.add_argument('--do_rl', action='store_true', default=False, help='Use RL to train PLSTM')

    parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--model', default='gpt2', type=str,
                        help="Pretrained model to use as encoder. "
                             "Use one of gpt2, roberta-base, bert-base-uncased")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=8, type=int, help="The batch size to use during training")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning Rate of adam")
    parser.add_argument('--warmup_steps', default=5000, type=int, help="Number of steps for increasing lr to max")
    parser.add_argument('--every', default=250, type=int, help="Log after every n examples")
    parser.add_argument('--save_every', default=2, type=int, help="Save after every n epochs")
    parser.add_argument('--resume_train', action='store_true', default=False, help="Resume from save model epoch")
    parser.add_argument('--save_base_dir', default='',
                        type=str, help="Save model and ckpt in this directory")
    parser.add_argument('--model_path', default='',
                        type=str, help="Load model from this path and train")
    parser.add_argument('--ckpt_path', default='',
                        type=str, help="Load checkpoint from this path")

    # val args
    parser.add_argument('--train_set', action='store_true', default=False,
                        help='Whether to find the coverage on the train set')
    parser.add_argument('--out_id', default='', type=str, help='Output id for storing model outputs')
    parser.add_argument('--max_actions', default=50, type=int, help="Max actions to consider while parsing")
    parser.add_argument('--dbg_sent', default='',
                        type=str, help="Run only on the given sentence for debugging")
    parser.add_argument('--ranker_model_path', default='',
                        type=str, help="Load ranker model from this path")
    parser.add_argument('--beam_width', default=10, type=int, help="Beam width to consider while decoding")
    parser.add_argument('--top_k', default=1, type=int, help="Number of best k programs to rank")

    # rl args
    # batch_size, max_actions,
    parser.add_argument('--episodes', default=10000, type=int, help="Number of episodes to train the model with RL")
    parser.add_argument('--avg_hist', default=100, type=int, help="Number of episodes to take avg of while logging")
    parser.add_argument('--value_coeff', default=0.5, type=float, help="Coeff for value loss")

    return parser.parse_args()


def val_all_plstm_models(args):
    import glob
    out_dict = {'Coverage': {}, 'FN': {}, 'FP': {}, 'TP': {}}
    for model_path in glob.glob(' /mnt/hhd/rohan/plstm_models/ws_model_*.pt'):
        epoch_idx = int(model_path[-6:-3])
        plstm_model = ProgramLSTM.load(model_path, True)
        plstm_model.to('CUDA')
        plstm_model.eval()
        with torch.no_grad():
            val_results = ProgramLSTM.get_model_results('data/plstm_valid.json', plstm_model, args)
        val_results = [(x[-1], x[-2] is not None) for x in val_results]
        ctr = Counter(val_results)
        ctr2 = {
            'FN': ctr[(False, False)],
            'FP': ctr[(False, True)],
            'TP': ctr[(True, True)],
        }
        coverage = ctr2['TP'] / len(val_results)
        out_dict['Coverage'][epoch_idx] = coverage
        out_dict['FN'][epoch_idx] = ctr2['FN']
        out_dict['FP'][epoch_idx] = ctr2['FP']
        out_dict['TP'][epoch_idx] = ctr2['TP']
    pprint.pprint(out_dict, indent=2)
    return


def main():
    # create_vocab()
    # test_program_tree()
    # inc_precision()
    args = init_plstm_arg_parser()
    if args.do_train:
        ProgramLSTM.train_program_lstm(args)
    elif args.do_rl:
        ProgramLSTM.train_rl_program_lstm(args)
    elif args.do_test:
        ProgramLSTM.test_program_lstm(args)
    return


if __name__ == '__main__':
    main()
