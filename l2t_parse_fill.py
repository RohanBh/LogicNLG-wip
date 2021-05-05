import copy
import json
import multiprocessing as mp
import os
import random
import re
import time

import nltk
import numpy as np
import pandas
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
from unidecode import unidecode

from APIs import non_triggers, fuzzy_match
from l2t_api import obj_compare, APIs, pat_add, pat_num, pat_month, ExeError, safe_obj_compare

with open('data/freq_list.json') as f:
    vocab = json.load(f)

with open('data/stop_words.json') as f:
    stop_words = json.load(f)

months_a = ['january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december']
months_b = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
a2b = {a: b for a, b in zip(months_a, months_b)}
b2a = {b: a for a, b in zip(months_a, months_b)}


class Node(object):
    def __init__(self, rows, memory_str, memory_num, memory_date, header_str,
                 header_num, header_date, must_have, must_not_have):
        # For intermediate results
        self.memory_str = memory_str
        self.memory_num = memory_num
        self.memory_date = memory_date
        self.header_str = header_str
        self.header_num = header_num
        self.header_date = header_date
        self.trace_str = [v for k, v in memory_str]
        self.trace_num = [v for k, v in memory_num]
        self.trace_date = [v for k, v in memory_date]
        # For intermediate data frame
        self.rows = [("all_rows", rows)]

        self.cur_str = ""
        self.cur_strs = []
        self.cur_funcs = []

        self.must_have = must_have
        self.must_not_have = must_not_have

        self.row_counter = [1]
        return

    def done(self):
        # if self.memory_str_len == 0 and self.memory_num_len == 0 and \
        #         all([_ > 0 for _ in self.row_counter]):
        finished_num = all(['tmp_input' == x[0] or 'tmp' not in x[0] for x in self.memory_num])
        finished_date = all(['tmp_input' == x[0] or 'tmp' not in x[0] for x in self.memory_date])
        finished_str = all(['tmp_input' == x[0] or 'tmp' not in x[0] for x in self.memory_str])
        if finished_num and finished_str and finished_date and all([_ > 0 for _ in self.row_counter]):
            for funcs in self.must_have:
                if any([f in self.cur_funcs for f in funcs]):
                    continue
                else:
                    return False
            return True
        else:
            return False

    def tostring(self):
        print("memory_str:", self.memory_str)
        print("memory_num:", self.memory_num)
        print("header_str:", self.header_str)
        print("header_num:", self.header_num)
        print("trace:", self.cur_str)

    def concat(self, new_str, k):
        func = new_str.split('(')[0]
        self.cur_funcs.append(func)
        self.cur_strs.append(new_str)

    def exist(self, command):
        return command in self.cur_strs

    def clone(self, command, k):
        tmp = copy.deepcopy(self)
        tmp.concat(command, k)
        return tmp

    @property
    def memory_str_len(self):
        return len(self.memory_str)

    @property
    def memory_num_len(self):
        return len(self.memory_num)

    @property
    def memory_date_len(self):
        return len(self.memory_date)

    @property
    def memory_len(self):
        return len(self.memory_num) + len(self.memory_str) + len(self.memory_date)

    @property
    def mem_object_len(self):
        return len(self.memory_num) + len(self.memory_date)

    @property
    def tmp_memory_num_len(self):
        return len([_ for _ in self.memory_num if "tmp_" in _ and _ not in ["tmp_none", 'tmp_input']])
        # return len(self.memory_num)

    @property
    def tmp_memory_str_len(self):
        return len([_ for _ in self.memory_str if "tmp_" in _])

    @property
    def row_num(self):
        return len(self.rows) - 1

    @property
    def hash(self):
        return hash(frozenset(self.cur_strs))

    @property
    def all_headers(self):
        return self.header_num + self.header_str + self.header_date

    @property
    def obj_headers(self):
        return self.header_num + self.header_date

    @property
    def memories(self):
        return self.memory_num + self.memory_str + self.memory_date

    @property
    def mem_objects(self):
        return self.memory_num + self.memory_date

    @property
    def all_traces(self):
        return self.trace_num + self.trace_str + self.trace_date

    @property
    def mem_obj_traces(self):
        return self.trace_num + self.trace_date

    def append_result(self, command, r):
        self.cur_str = "{}={}".format(command, r)

    def get_memory_type(self, header):
        # if i < self.memory_num_len:
        #     return 'num'
        # if i - self.memory_num_len < self.memory_str_len:
        #     return 'str'
        # if i - self.memory_num_len - self.memory_str_len < self.memory_date_len:
        #     return 'date'
        # raise ValueError(f'Index out of bounds for memories: {i}')
        if any(x[0] == header for x in self.memory_num) or any(x == header for x in self.header_num):
            return 'num'
        if any(x[0] == header for x in self.memory_str) or any(x == header for x in self.header_str):
            return 'str'
        if any(x[0] == header for x in self.memory_date) or any(x == header for x in self.header_date):
            return 'date'
        if header == 'tmp_count':
            return 'num'
        if 'tmp_' in header:
            return self.get_memory_type(header.replace('tmp_', ''))
        raise ValueError(f"Unseen header: {header}")

    def add_memory_num(self, header, val, command):
        try:
            val = float(val)
        except:
            try:
                val = int(val)
            except:
                pass
        if type(val) == type(1) or type(val) == type(1.2):
            val = val
        elif type(val) is pd.Series:
            val = val.item()

        self.memory_num.append((header, val))
        self.trace_num.append(command)

    def add_memory_str(self, header, val, command):
        if isinstance(val, str):
            self.memory_str.append((header, val))
            self.trace_str.append(command)
        # else:
        #     raise ValueError("type error: {}".format(type(val)))

    def add_memory_date(self, header, val, command):
        if isinstance(val, str) and len(re.findall(pat_month, val)) > 0:
            self.memory_date.append((header, val))
            self.trace_date.append(command)
        # else:
        #     raise ValueError(f"type error: {type(val)}, value: {val}")

    def add_memory(self, header, val, command, aux_header=None):
        if aux_header is None:
            aux_header = header
        mem_type = self.get_memory_type(aux_header)
        if mem_type == 'num':
            return self.add_memory_num(header, val, command)
        if mem_type == 'str':
            return self.add_memory_str(header, val, command)
        if mem_type == 'date':
            return self.add_memory_date(header, val, command)
        return

    def add_header_str(self, header):
        self.header_str.append(header)

    def add_header_num(self, header):
        self.header_num.append(header)

    def add_header_date(self, header):
        self.header_date.append(header)

    def add_rows(self, header, val):
        if isinstance(val, pandas.DataFrame):
            # for row_h, row in self.rows:
            #    if len(row) == len(val) and row.iloc[0][0] == val.iloc[0][0]:
            #        return
            if any([row_h == header for row_h, row in self.rows]):
                return
            self.rows.append((header, val.reset_index(drop=True)))
            self.row_counter.append(0)
        else:
            raise ValueError("type error")

    def inc_row_counter(self, i):
        self.row_counter[i] += 1

    def delete_memory_num(self, *args):
        new_mem_num = []
        new_trace_num = []
        for k in range(len(self.memory_num)):
            if k in args:
                continue
            else:
                new_mem_num.append(self.memory_num[k])
                new_trace_num.append(self.trace_num[k])

        self.memory_num = new_mem_num
        self.trace_num = new_trace_num

    def delete_memory_str(self, *args):
        new_mem_str = []
        new_trace_str = []
        for k in range(len(self.memory_str)):
            if k in args:
                continue
            else:
                new_mem_str.append(self.memory_str[k])
                new_trace_str.append(self.trace_str[k])

        self.memory_str = new_mem_str
        self.trace_str = new_trace_str

    def delete_memory_date(self, *args):
        new_mem_date = []
        new_trace_date = []
        for k in range(len(self.memory_date)):
            if k in args:
                continue
            else:
                new_mem_date.append(self.memory_date[k])
                new_trace_date.append(self.trace_date[k])

        self.memory_date = new_mem_date
        self.trace_date = new_trace_date

    def delete_memory(self, h, va):
        mem_type = self.get_memory_type(h)
        if mem_type == 'num':
            return self.delete_memory_num(self.memory_num.index((h, va)))
        if mem_type == 'str':
            return self.delete_memory_str(self.memory_str.index((h, va)))
        if mem_type == 'date':
            return self.delete_memory_date(self.memory_date.index((h, va)))
        return


class FailedCommand:
    def __init__(self, message):
        self.message = message


def dynamic_programming(name, t, orig_sent, sent, tags, mem_str, mem_num, mem_date, head_str,
                        head_num, head_date, masked_val, num=6, debug=False):
    masked_val = (masked_val[0], str(masked_val[1]))
    must_have = []
    must_not_have = []
    # Goes through non_triggers and absence of any triggers marks those
    # functions as inactive. This is useful for search-space pruning
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
            if not all(flags):
                must_not_have.append(k)
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
            if not flag:
                must_not_have.append(k)

    node = Node(memory_str=mem_str, memory_num=mem_num, memory_date=mem_date, rows=t,
                header_str=head_str, header_num=head_num, header_date=head_date,
                must_have=must_have, must_not_have=must_not_have)

    # Whether a count function should be invoked on all rows?
    count_all = any([k == 'tmp_input' for k, v in mem_num])

    # The result storage
    finished = []
    hist = [[node]] + [[] for _ in range(num)]
    cache = {}

    def call(command, f, *args):
        if command not in cache:
            try:
                cache[command] = f(*args)
            except ExeError as err:
                cache[command] = FailedCommand(err.message)
            return cache[command]
        else:
            return cache[command]

    start_time = time.time()
    for step in range(len(hist) - 1):
        # Iterate over father nodes
        saved_hash = []

        def conditional_add(tmp, path):
            success = False
            if tmp.hash not in saved_hash:
                path.append(tmp)
                saved_hash.append(tmp.hash)
                success = True
            return success

        for root in hist[step]:
            # Iterate over API
            for k, v in APIs.items():

                # propose candidates
                if k in root.must_not_have:
                    continue

                if v['output'] == 'row' and root.row_num >= 2:
                    continue
                if v['output'] == 'num' and root.tmp_memory_num_len >= 3:
                    continue
                if v['output'] == 'str' and root.tmp_memory_str_len >= 3:
                    continue
                if 'inc_' in k and 'inc' in root.cur_funcs:
                    continue
                if v['output'] == 'bool':
                    continue

                # Incrementing/Decrementing
                if v['argument'] == ["any"]:
                    for i, (h, va) in enumerate(root.memories):
                        if v['output'] == 'any':
                            if step == 0 and "tmp" in h:
                                command = v['tostr'](root.all_traces[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    if isinstance(returned, FailedCommand):
                                        continue
                                    if tmp.done():
                                        continue
                                    else:
                                        tmp.add_memory(h, returned, returned)
                                        conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'none':
                            if step == 0 and "tmp" in h:
                                command = v['tostr'](root.all_traces[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory(h, va)
                                    if tmp.done():
                                        continue
                                    else:
                                        conditional_add(tmp, hist[step + 1])
                        else:
                            raise ValueError("Returned Type Wrong")

                # Count
                elif v['argument'] == ['row']:
                    for j, (row_h, row) in enumerate(root.rows):
                        if k == "count":
                            if row_h.startswith('filter'):
                                pass
                            elif row_h == "all_rows":
                                if count_all:
                                    pass
                                else:
                                    continue
                        else:
                            raise NotImplementedError(f"Unk func {k} in APIs")
                        command = v['tostr'](row_h)
                        if not root.exist(command):
                            tmp = root.clone(command, k)
                            tmp.inc_row_counter(j)
                            returned = call(command, v['function'], row)
                            if isinstance(returned, FailedCommand):
                                continue
                            if v['output'] == 'num':
                                if tmp.done():
                                    tmp.append_result(
                                        command,
                                        f'{returned}/' + str(obj_compare(masked_val[1], returned)))
                                    finished.append((tmp, returned))
                                else:
                                    tmp.add_memory_num("tmp_count", returned, command)
                                    conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")

                # hop, max, min, argmax, argmin,
                # [most/all]_[greater/less]_[eq]_inv, [most/all]_eq_inv
                elif v['argument'] == ['row', 'header']:
                    if "hop" in k:
                        # hop is special. It hops on all types of headers
                        # TODO: Refactor this. Change the signature of hop to 'row, 'header_any'
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.all_headers)):
                                command = v['tostr'](row_h, root.all_headers[l])
                                if "; " + root.all_headers[l] + ";" in row_h:
                                    continue
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.all_headers[l])
                                    if isinstance(returned, FailedCommand):
                                        continue
                                    if tmp.done():
                                        tmp.append_result(
                                            command,
                                            f'{returned}/' + str(obj_compare(masked_val[1], returned)))
                                        finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory("tmp_" + root.all_headers[l], returned, command,
                                                       root.all_headers[l])
                                        conditional_add(tmp, hist[step + 1])
                    else:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) == 1:
                                continue
                            for l in range(len(root.obj_headers)):
                                command = v['tostr'](row_h, root.obj_headers[l])
                                if not root.exist(command):
                                    returned = call(command, v['function'], row, root.obj_headers[l])
                                    if isinstance(returned, FailedCommand):
                                        continue

                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    if v['output'] == 'obj':
                                        if tmp.done():
                                            tmp.append_result(
                                                command,
                                                f'{returned}/' + str(obj_compare(masked_val[1], returned)))
                                            finished.append((tmp, returned))
                                        else:
                                            tmp.add_memory("tmp_" + root.obj_headers[l], returned, command,
                                                           root.obj_headers[l])
                                            conditional_add(tmp, hist[step + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'pair_obj':
                                        if tmp.done():
                                            if 'greater' in k:
                                                comp_type = "less_eq"
                                            elif 'less' in k:
                                                comp_type = "greater_eq"
                                            elif k == 'all_eq_inv':
                                                comp_type = "eq"
                                            else:
                                                raise ValueError(f"error, output of scope {k}")
                                            if len(re.findall(pat_month, masked_val[1])) > 0:
                                                if returned[1] is None:
                                                    continue
                                                tmp.append_result(
                                                    command,
                                                    f'{returned[1]}/' +
                                                    str(safe_obj_compare(masked_val[1], returned[1], type=comp_type)))
                                                finished.append((tmp, returned[1]))
                                            elif len(re.findall(pat_num, masked_val[1])) > 0:
                                                if returned[0] is None:
                                                    continue
                                                tmp.append_result(
                                                    command,
                                                    f'{returned[0]}/' +
                                                    str(safe_obj_compare(masked_val[1], returned[0], type=comp_type)))
                                                finished.append((tmp, returned[0]))
                                            else:
                                                continue
                                    elif v['output'] == 'pair_list_obj':
                                        if tmp.done():
                                            ix = None
                                            if len(re.findall(pat_month, masked_val[1])) > 0:
                                                ix = 1
                                            elif len(re.findall(pat_num, masked_val[1])) > 0:
                                                ix = 0
                                            if ix is not None:
                                                if returned[ix] is None:
                                                    continue
                                                for ret_val in returned[ix]:
                                                    tmp_new = copy.deepcopy(tmp)
                                                    tmp_new.append_result(
                                                        command,
                                                        f'{ret_val}/' + str(obj_compare(masked_val[1], ret_val)))
                                                    finished.append((tmp_new, ret_val))
                                            else:
                                                continue
                                    else:
                                        raise ValueError("error, output of scope")

                # avg, sum
                elif v['argument'] == ['row', 'header_num']:
                    for j, (row_h, row) in enumerate(root.rows):
                        if len(row) == 1:
                            continue
                        for l in range(len(root.header_num)):
                            command = v['tostr'](row_h, root.header_num[l])
                            if not root.exist(command):
                                tmp = root.clone(command, k)
                                tmp.inc_row_counter(j)

                                returned = call(command, v['function'], row, root.header_num[l])
                                if isinstance(returned, FailedCommand):
                                    continue
                                if v['output'] == 'num':
                                    if tmp.done():
                                        tmp.append_result(
                                            command,
                                            f'{returned}/' + str(obj_compare(masked_val[1], returned, True)))
                                        finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                else:
                                    raise ValueError("error, output of scope")

                # [most/all]_str_eq_inv
                elif v['argument'] == ['row', 'header_str']:
                    for j, (row_h, row) in enumerate(root.rows):
                        if len(row) == 1:
                            continue
                        for l in range(len(root.header_str)):
                            command = v['tostr'](row_h, root.header_str[l])
                            if not root.exist(command):
                                returned = call(command, v['function'], row, root.header_str[l])
                                if isinstance(returned, FailedCommand):
                                    continue
                                tmp = root.clone(command, k)
                                tmp.inc_row_counter(j)

                                if v['output'] == 'str':
                                    if tmp.done():
                                        tmp.append_result(
                                            command,
                                            f'{returned}/' + str(obj_compare(masked_val[1], returned)))
                                        finished.append((tmp, returned))
                                elif v['output'] == 'list_str':
                                    if tmp.done():
                                        for ret_val in returned:
                                            tmp_new = copy.deepcopy(tmp)
                                            tmp_new.append_result(
                                                command,
                                                f'{ret_val}/' + str(obj_compare(masked_val[1], ret_val)))
                                            finished.append((tmp_new, ret_val))
                                else:
                                    raise ValueError("error, output of scope")

                # diff
                elif v['argument'] == ['obj', 'obj']:
                    if root.mem_object_len < 2:
                        continue
                    for l in range(0, root.mem_object_len - 1):
                        for m in range(l + 1, root.mem_object_len):
                            if root.mem_objects[l][0] == 'ntharg' or root.mem_objects[m][0] == 'ntharg':
                                continue
                            if 'tmp_' in root.mem_objects[l][0] or 'tmp_' in root.mem_objects[m][0]:
                                if ("tmp_input" == root.mem_objects[l][0] and "tmp_" not in root.mem_objects[m][0]) or \
                                        ("tmp_input" == root.mem_objects[m][0] and "tmp_" not in root.mem_objects[l][
                                            0]):
                                    continue
                                elif root.mem_objects[l][0] == root.mem_objects[m][0] == "tmp_input":
                                    continue
                            else:
                                continue

                            type_l = root.mem_objects[l][0].replace('tmp_', '')
                            type_m = root.mem_objects[m][0].replace('tmp_', '')
                            if v['output'] == 'obj':
                                if type_l == type_m:
                                    # Two direction:
                                    for dir1, dir2 in zip([l, m], [m, l]):
                                        command = v['tostr'](root.mem_obj_traces[dir1], root.mem_obj_traces[dir2])
                                        if not root.exist(command):
                                            tmp = root.clone(command, k)
                                            tmp.delete_memory(*root.mem_objects[l])
                                            tmp.delete_memory(*root.mem_objects[m])
                                            returned = call(command, v['function'],
                                                            root.mem_objects[l][1], root.mem_objects[l][1])
                                            if isinstance(returned, FailedCommand):
                                                continue
                                            if tmp.done():
                                                tmp.append_result(
                                                    command,
                                                    f'{returned}/' + str(obj_compare(masked_val[1], returned)))
                                                finished.append((tmp, returned))
                                            else:
                                                tmp.add_memory("tmp_" + root.memory_num[dir1][0], returned,
                                                               command, root.memory_num[dir1][0])
                                                conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, output of scope")

                # filter_str_eq or filter_str_not_eq
                elif v['argument'] == ['row', 'header_str', 'str']:
                    for j, (row_h, row) in enumerate(root.rows):
                        # It does not make sense to do filter operation on one row
                        if len(row) == 1:
                            continue
                        for i, (h, va) in enumerate(root.memory_str):
                            if "tmp_" not in h and h != 'ntharg':
                                command = v['tostr'](row_h, h, root.trace_str[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_str(tmp.memory_str.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if isinstance(returned, FailedCommand):
                                        continue
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    else:
                                        raise ValueError('error, output of scope')

                # filter_eq
                elif v['argument'] == ['row', 'header', 'obj']:
                    for j, (row_h, row) in enumerate(root.rows):
                        # It does not make sense to do filter operation on one row
                        if len(row) == 1:
                            continue
                        for i, (h, va) in enumerate(root.mem_objects):
                            if "tmp_" not in h and h != 'ntharg':
                                command = v['tostr'](row_h, h, root.mem_obj_traces[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory(h, va)

                                    returned = call(command, v['function'], row, h, va)
                                    if isinstance(returned, FailedCommand):
                                        continue
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    else:
                                        raise ValueError('error, output of scope')

                # nth_argmax, nth_max
                elif v['argument'] == ['row', 'header', 'num']:
                    for j, (row_h, row) in enumerate(root.rows):
                        # It does not make sense to do max/argmax operation on one row
                        if len(row) == 1:
                            continue
                        for l in range(len(root.obj_headers)):
                            for _h, va in root.memory_num:
                                if _h != 'ntharg':
                                    continue
                                command = v['tostr'](row_h, root.obj_headers[l], va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)

                                    returned = call(command, v['function'], row, root.obj_headers[l], va)
                                    if isinstance(returned, FailedCommand):
                                        continue
                                    if v['output'] == 'obj':
                                        if tmp.done():
                                            tmp.append_result(
                                                command,
                                                f'{returned}/' + str(obj_compare(masked_val[1], returned)))
                                            finished.append((tmp, returned))
                                        else:
                                            tmp.add_memory("tmp_" + root.obj_headers[l], returned, command,
                                                           root.obj_headers[l])
                                            conditional_add(tmp, hist[step + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    else:
                                        raise ValueError("error, output of scope")

                # greater_str_inv, less_str_inv
                elif v['argument'] == ['row', 'header_str', 'str', 'header']:
                    # TODO: Refactor. Change the signature to header_any instead of header_str
                    for j, (row_h, row) in enumerate(root.rows):
                        # It does not make sense to do greater_inv, less_inv operation on one row
                        if len(row) == 1:
                            continue
                        for i, (h1, va1) in enumerate(root.memories):
                            if "tmp_" in h1 or h1 == 'ntharg':
                                continue
                            for l in range(len(root.obj_headers)):
                                command = v['tostr'](row_h, h1, va1, root.obj_headers[l])
                                if not root.exist(command):
                                    returned = call(command, v['function'], row, h1, va1, root.obj_headers[l])
                                    if isinstance(returned, FailedCommand):
                                        continue
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_str(tmp.memories.index((h1, va1)))

                                    if v['output'] == 'list_str':
                                        if tmp.done():
                                            for ret_val in returned:
                                                tmp_new = copy.deepcopy(tmp)
                                                tmp_new.append_result(
                                                    command,
                                                    f'{ret_val}/' + str(obj_compare(masked_val[1], ret_val)))
                                                finished.append((tmp_new, ret_val))
                                    else:
                                        raise ValueError('error, output of scope')

                else:
                    continue

        if len(finished) > 100 or time.time() - start_time > 30:
            break

    return (name, orig_sent, sent, [_[0].cur_str for _ in finished])


def augment(s):
    recover_dict = {}
    if 'first' in s:
        s.append("1st")
        recover_dict[s[-1]] = 'first'
    elif 'second' in s:
        s.append("2nd")
        recover_dict[s[-1]] = 'second'
    elif 'third' in s:
        s.append("3rd")
        recover_dict[s[-1]] = 'third'
    elif 'fourth' in s:
        s.append("4th")
        recover_dict[s[-1]] = 'fourth'
    elif 'fifth' in s:
        s.append("5th")
        recover_dict[s[-1]] = 'fifth'
    elif 'sixth' in s:
        s.append("6th")
        recover_dict[s[-1]] = 'sixth'
    elif 'seventh' in s:
        s.append("7th")
        recover_dict[s[-1]] = 'seventh'
    elif 'eighth' in s:
        s.append("8th")
        recover_dict[s[-1]] = 'eighth'
    elif 'ninth' in s:
        s.append("9th")
        recover_dict[s[-1]] = 'ninth'
    elif 'tenth' in s:
        s.append("10th")
        recover_dict[s[-1]] = 'tenth'
    elif 'eleventh' in s:
        s.append("11th")
        recover_dict[s[-1]] = 'eleventh'
    elif 'twelfth' in s:
        s.append("12th")
        recover_dict[s[-1]] = 'twelfth'
    elif 'thirteenth' in s:
        s.append("13th")
        recover_dict[s[-1]] = 'thirteenth'
    elif 'fourteenth' in s:
        s.append("14th")
        recover_dict[s[-1]] = 'fourteenth'
    elif 'fifteenth' in s:
        s.append("15th")
        recover_dict[s[-1]] = 'fifteenth'

    for i in range(1, 10):
        if "0" + str(i) in s:
            s.append(str(i))
            recover_dict[s[-1]] = "0" + str(i)

    if 'crowd' in s or 'attendance' in s:
        s.append("people")
        recover_dict[s[-1]] = 'crowd'
        s.append("audience")
        recover_dict[s[-1]] = 'crowd'

    if any([_ in months_a + months_b for _ in s]):
        for i in range(1, 32):
            if str(i) in s:
                if i % 10 == 1:
                    s.append(str(i) + "st")
                elif i % 10 == 2:
                    s.append(str(i) + "nd")
                elif i % 10 == 3:
                    s.append(str(i) + "rd")
                else:
                    s.append(str(i) + "th")
                recover_dict[s[-1]] = str(i)

        for k in a2b:
            if k in s:
                s.append(a2b[k])
                recover_dict[s[-1]] = k

        for k in b2a:
            if k in s:
                s.append(b2a[k])
                recover_dict[s[-1]] = k

    return s, recover_dict


def replace_useless(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    s = s.replace('/', '')
    s = s.replace('  ', '')
    return s


def get_closest(inp, string, indexes, tabs, threshold):
    if string in stop_words:
        return None

    dist = 10000
    rep_string = replace_useless(string)
    len_string = len(rep_string.split())

    minimum = []
    for index in indexes:
        entity = replace_useless(tabs[index[0]][index[1]])
        len_tab = len(entity.split())
        if abs(len_tab - len_string) < dist:
            minimum = [index]
            dist = abs(len_tab - len_string)
        elif abs(len_tab - len_string) == dist:
            minimum.append(index)

    vocabs = []
    for s in rep_string.split(' '):
        vocabs.append(vocab.get(s, 10000))

    # Whether contain rare words
    if dist == 0:
        return minimum[0]

    # String Length
    feature = [len_string]
    # Proportion
    feature.append(-dist / (len_string + dist + 0.) * 4)
    if any([(s.isdigit() and int(s) < 100) for s in rep_string.split()]):
        feature.extend([0, 0])
    else:
        # Quite rare words
        if max(vocabs) > 800:
            feature.append(1)
        else:
            feature.append(-1)
        # Whether contain super rare words
        if max(vocabs) > 2000:
            feature.append(3)
        else:
            feature.append(0)
    # Whether it is only a word
    if len_string > 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether candidate has only one
    if len(indexes) == 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether cover over half of it
    if len_string > dist:
        feature.append(1)
    else:
        feature.append(0)

    # Whether contains alternative
    cand = replace_useless(tabs[minimum[0][0]][minimum[0][1]])
    if '(' in cand and ')' in cand:
        feature.append(2)
    else:
        feature.append(0)
    # Match more with the header
    if minimum[0][0] == 0:
        feature.append(2)
    else:
        feature.append(0)
    # Whether it is a month
    if any([" " + _ + " " in " " + rep_string + " " for _ in months_a + months_b]):
        feature.append(5)
    else:
        feature.append(0)

    # Whether it matches against the candidate
    if rep_string in cand:
        feature.append(0)
    else:
        feature.append(-5)

    if sum(feature) > threshold:
        if len(minimum) > 1:
            if minimum[0][0] > 0:
                return [-2, minimum[0][1]]
            else:
                return minimum[0]
        else:
            return minimum[0]
    else:
        return None


def replace_number(string):
    string = re.sub(r'(\b)zero(\b)', r'\g<1>0\g<2>', string)
    string = re.sub(r'(\b)one(\b)', r'\g<1>1\g<2>', string)
    string = re.sub(r'(\b)two(\b)', '\g<1>2\g<2>', string)
    string = re.sub(r'(\b)three(\b)', '\g<1>3\g<2>', string)
    string = re.sub(r'(\b)four(\b)', '\g<1>4\g<2>', string)
    string = re.sub(r'(\b)five(\b)', '\g<1>5\g<2>', string)
    string = re.sub(r'(\b)six(\b)', '\g<1>6\g<2>', string)
    string = re.sub(r'(\b)seven(\b)', '\g<1>7\g<2>', string)
    string = re.sub(r'(\b)eight(\b)', '\g<1>8\g<2>', string)
    string = re.sub(r'(\b)nine(\b)', '\g<1>9\g<2>', string)
    string = re.sub(r'(\b)ten(\b)', '\g<1>10\g<2>', string)
    string = re.sub(r'(\b)eleven(\b)', '\g<1>11\g<2>', string)
    string = re.sub(r'(\b)twelve(\b)', '\g<1>12\g<2>', string)
    string = re.sub(r'(\b)thirteen(\b)', '\g<1>13\g<2>', string)
    string = re.sub(r'(\b)fourteen(\b)', '\g<1>14\g<2>', string)
    string = re.sub(r'(\b)fifteen(\b)', '\g<1>15\g<2>', string)
    string = re.sub(r'(\b)sixteen(\b)', '\g<1>16\g<2>', string)
    string = re.sub(r'(\b)seventeen(\b)', '\g<1>17\g<2>', string)
    string = re.sub(r'(\b)eighteen(\b)', '\g<1>18\g<2>', string)
    string = re.sub(r'(\b)nineteen(\b)', '\g<1>19\g<2>', string)
    string = re.sub(r'(\b)twenty(\b)', '\g<1>20\g<2>', string)

    string = string.replace('a hundred', '100')
    string = string.replace('a thousand', '1000')

    string = re.sub(r'\b([0-9]+) hundred', r'\g<1>00', string)
    string = re.sub(r'\b([0-9]+) thousand', r'\g<1>000', string)

    return string


def replace(w, transliterate):
    if w in transliterate:
        return transliterate[w]
    else:
        return w


def recover(buf, recover_dict, content):
    if len(recover_dict) == 0:
        return buf
    else:
        new_buf = []
        for w in buf.split(' '):
            # Changed `from w not in content` prevents cases like get not in '... gets ...'. get should be recovered
            if w not in content.split(' '):
                new_buf.append(recover_dict.get(w, w))
            else:
                new_buf.append(w)
        return ' '.join(new_buf)


def postprocess(inp, pos_tags, backbone, trans_backbone, transliterate, tabs, recover_dicts, repeat, threshold=1.0):
    # This function is the actual entity linker. It goes through the input word by word. If the input token is found
    # in the backbone (or transbb), it creates a buffer and keeps on filling it with consecutive tokens in the sentence
    # until: 1) The next token is not in the table/backbone 2) The next token makes the backbone intersection
    # (set of candidate cells) an empty set 3) The sentence ends
    # Now, with the buffer just before this event is passed to get_closest function which matches the cell
    # with the closest length as that of the buffer. If the lengths match then any one of the matching length
    # cells are returned. Else, in case of a mismatch, it creates a feature vector for the buffered tokens and if the
    # sum of the features is above a threshold, it uses one of the matches to return a cell.
    new_str = []
    new_tags = []
    buf = ""
    pos_buf = []
    last = set()
    prev_closest = []

    for w, p in zip(inp, pos_tags):
        if (w in backbone) and (
                (" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            if buf == "":
                last = set(backbone[w])
                buf = w
                pos_buf.append(p)
            else:
                proposed = set(backbone[w]) & last
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])

                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = w
                    last = set(backbone[w])
                    pos_buf.append(p)
                else:
                    last = proposed
                    buf += " " + w
                    pos_buf.append(p)

        elif w in trans_backbone and (
                (" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            if buf == "":
                last = set(trans_backbone[w])
                buf = transliterate[w]
                pos_buf.append(p)
            else:
                proposed = set(trans_backbone[w]) & last
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])
                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = transliterate[w]
                    last = set(trans_backbone[w])
                    pos_buf.append(p)
                else:
                    buf += " " + transliterate[w]
                    last = proposed
                    pos_buf.append(p)

        else:
            if buf != "":
                closest = get_closest(inp, buf, last, tabs, threshold)
                if closest:
                    buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                      tabs[closest[0]][closest[1]]), closest[0], closest[1])
                new_str.append(buf)
                if buf.startswith("#"):
                    new_tags.append('ENT')
                else:
                    new_tags.extend(pos_buf)
                pos_buf = []

            buf = ""
            last = set()
            new_str.append(w)
            new_tags.append(p)

    if buf != "":
        closest = get_closest(inp, buf, last, tabs, threshold)
        if closest:
            buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                              tabs[closest[0]][closest[1]]), closest[0], closest[1])
        new_str.append(buf)
        if buf.startswith("#"):
            new_tags.append('ENT')
        else:
            new_tags.extend(pos_buf)
        pos_buf = []

    return " ".join(new_str), " ".join(new_tags)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def isnumber(string):
    return string in [np.dtype('int64'), np.dtype('int32'), np.dtype('float32'), np.dtype('float64')]


def split(string, option):
    if option == "row":
        return string.split(',')[0]
    else:
        return string.split(',')[1]


def merge_strings(string, tags=None):
    buff = ""
    inside = False
    words = []

    for c in string:
        if c == "#" and not inside:
            inside = True
            buff += c
        elif c == "#" and inside:
            inside = False
            buff += c
            words.append(buff)
            buff = ""
        elif c == " " and not inside:
            if buff:
                words.append(buff)
            buff = ""
        elif c == " " and inside:
            buff += c
        else:
            buff += c

    if buff:
        words.append(buff)

    tags = tags.split(' ')
    assert len(words) == len(tags), "{} and {}".format(words, tags)

    i = 0
    while i < len(words):
        if i < 2:
            i += 1
        elif words[i].startswith('#') and (not words[i - 1].startswith('#')) and words[i - 2].startswith('#'):
            if is_number(words[i].split(';')[0][1:]) and is_number(words[i - 2].split(';')[0][1:]):
                i += 1
            else:
                prev_idx = words[i - 2].split(';')[1][:-1].split(',')
                cur_idx = words[i].split(';')[1][:-1].split(',')
                if cur_idx == prev_idx or (prev_idx[0] == '-2' and prev_idx[1] == cur_idx[1]):
                    position = "{},{}".format(cur_idx[0], cur_idx[1])
                    candidate = words[i - 2].split(';')[0] + " " + words[i].split(';')[0][1:] + ";" + position + "#"
                    words[i] = candidate
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                else:
                    i += 1
        else:
            i += 1

    return " ".join(words), " ".join(tags)


def get_col_types(t):
    col2type = {}
    for i, col in enumerate(t.columns):
        if isnumber(t[col].dtype):
            col2type[i] = 'num'
            continue
        col_series = t[col].astype('str')
        pats = col_series.str.extract(pat_add, expand=False)
        if pats.isnull().all():
            pats = col_series.str.extract(pat_num, expand=False)
        if not pats.isnull().all():
            col2type[i] = 'num'
            continue
        date_pats = col_series.str.extract(pat_month, expand=False)
        if not date_pats.isnull().all():
            col2type[i] = 'date'
            continue
        col2type[i] = 'str'

    return col2type


class Parser(object):
    def __init__(self, folder, lemmatize_verbs=True):
        self.folder = folder

        with open('data/table_to_page.json') as f:
            files = json.load(f)
            self.title_mapping = {k: v[0] for k, v in files.items()}

        if lemmatize_verbs:
            self.tag_dict = {"JJ": wordnet.ADJ,
                             "NN": wordnet.NOUN,
                             "NNS": wordnet.NOUN,
                             "NNP": wordnet.NOUN,
                             "NNPS": wordnet.NOUN,
                             "VB": wordnet.VERB,
                             "VBD": wordnet.VERB,
                             "VBG": wordnet.VERB,
                             "VBN": wordnet.VERB,
                             "VBP": wordnet.VERB,
                             "VBZ": wordnet.VERB,
                             "RB": wordnet.ADV,
                             "RP": wordnet.ADV}
        else:
            self.tag_dict = {"JJ": wordnet.ADJ,
                             "NN": wordnet.NOUN,
                             "NNS": wordnet.NOUN,
                             "NNP": wordnet.NOUN,
                             "NNPS": wordnet.NOUN,
                             "RB": wordnet.ADV,
                             "RP": wordnet.ADV}

        self.lemmatizer = WordNetLemmatizer()
        self.model1 = None
        self.model2 = None
        self.optimizer = None
        self._embeddings_dict = {}
        return

    @property
    def embeddings_dict(self):
        if len(self._embeddings_dict) > 0:
            return self._embeddings_dict
        with open("data/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self._embeddings_dict[word] = vector
        return self._embeddings_dict

    def get_embd(self, w):
        w = str(w)
        return np.average(list(self.embeddings_dict.get(_w, np.zeros(50)) for _w in w.split(' ')), axis=0)

    def get_lemmatize(self, words, return_pos):
        recover_dict = {}
        words = nltk.word_tokenize(words)
        words = [word for word in words if word]
        pos_tags = [_[1] for _ in nltk.pos_tag(words)]
        word_roots = []
        for w, p in zip(words, pos_tags):
            if is_ascii(w) and p in self.tag_dict:
                lemm = self.lemmatizer.lemmatize(w, self.tag_dict[p])
                if lemm != w:
                    recover_dict[lemm] = w
                word_roots.append(lemm)
            else:
                word_roots.append(w)
        if return_pos:
            return word_roots, recover_dict, pos_tags
        else:
            return word_roots, recover_dict

    def get_table(self, table_name):
        file = os.path.join(self.folder, table_name)
        table = pandas.read_csv(file, delimiter="#")
        return table

    def entity_link(self, table_name, lemmatized_sent, pos_tags):
        # Maps all types of tokens (normal & accent word) in the tables to their positions
        backbone = {}
        # Like backbone: But only maps the uni-decoded word (accents removed) to their pos
        trans_backbone = {}
        # Keeps additional mapping - uni-decoded (accent removed on chars) to non-ascii word (with accents) mapping
        transliterate = {}
        # Keeps the table in an arr. tabs[-1] is the table title
        tabs = []
        # Used to get the word back from the lemmatized word
        recover_dicts = []
        # Contains tokens that occur multiple times in a cell except months
        repeat = set()
        table_title = self.title_mapping[table_name]

        with open(os.path.join(self.folder, table_name), 'r') as f:
            for k, _ in enumerate(f.readlines()):
                tabs.append([])
                recover_dicts.append([])
                for l, w in enumerate(_.strip().split('#')):
                    tabs[-1].append(w)
                    if len(w) > 0:
                        lemmatized_w, recover_dict = self.get_lemmatize(w, False)
                        lemmatized_w, new_dict = augment(lemmatized_w)
                        recover_dict.update(new_dict)
                        recover_dicts[-1].append(recover_dict)
                        for i, sub in enumerate(lemmatized_w):
                            if sub not in backbone:
                                backbone[sub] = [(k, l)]
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)] = [(k, l)]
                                    transliterate[unidecode(sub)] = sub
                            else:
                                if (k, l) not in backbone[sub]:
                                    backbone[sub].append((k, l))
                                else:
                                    if sub not in months_a + months_b:
                                        repeat.add(sub)
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)].append((k, l))
                                    transliterate[unidecode(sub)] = sub

                        for i, sub in enumerate(w.split(' ')):
                            if sub not in backbone:
                                backbone[sub] = [(k, l)]
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)] = [(k, l)]
                                    transliterate[unidecode(sub)] = sub
                            else:
                                if (k, l) not in backbone[sub]:
                                    backbone[sub].append((k, l))
                                if not is_ascii(sub):
                                    trans_backbone[unidecode(sub)].append((k, l))
                                    transliterate[unidecode(sub)] = sub
                    else:
                        recover_dicts[-1].append({})

        # Masking the caption
        captions, _ = self.get_lemmatize(table_title.strip(), False)
        for i, w in enumerate(captions):
            if w not in backbone:
                backbone[w] = [(-1, -1)]
            else:
                backbone[w].append((-1, -1))

        tabs.append([" ".join(captions)])

        results = []

        sent, tags = postprocess(lemmatized_sent, pos_tags, backbone, trans_backbone,
                                 transliterate, tabs, recover_dicts, repeat, threshold=1.0)

        if re.search(r',[0-9]+#', sent):
            sent, tags = postprocess(lemmatized_sent, pos_tags, backbone, trans_backbone,
                                     transliterate, tabs, recover_dicts, repeat, threshold=0.0)

        sent, tags = merge_strings(sent, tags)

        return sent, tags

    def get_table2vecs(self, table_name):
        table_title = self.title_mapping[table_name]
        t = self.get_table(table_name)
        idx2vec = {}
        col2type = get_col_types(t)

        for j, col in enumerate(t.columns):
            features = []
            if col2type[j] == 'num':
                features.append(0)
            elif col2type[j] == 'date':
                features.append(1)
            else:
                features.append(2)
            features.append(0)
            features.append(j)
            features.extend(np.average(list(self.get_embd(w) for w in t[col][:3]), axis=0))
            idx2vec[(0, j)] = [self.get_embd(w) for w in col.split(' ')] + [features]

        for i in range(t.shape[0]):
            for j, col in enumerate(t.columns):
                val = str(t.iloc[i, j])
                features = []
                if col2type[j] == 'num':
                    features.append(0)
                elif col2type[j] == 'date':
                    features.append(1)
                else:
                    features.append(2)
                features.append(i)
                features.append(j)
                nb_range = list(range(max(i - 2, 0), i)) + list(range(i + 1, min(t.shape[0] - 1, i + 2) + 1))
                features.extend(np.average(list(self.get_embd(t.iloc[_i][j]) for _i in nb_range), axis=0))
                idx2vec[(i + 1, j)] = [self.get_embd(w) for w in val.split(' ')] + [features]

        features = [2, -1, -1] + [0] * 50
        idx2vec[(-1, -1)] = [self.get_embd(w) for w in table_title.split(' ')] + [features]
        return idx2vec

    def new_entity_link(self, table_name, sent, pos):
        import torch
        import torch.nn.functional as F
        from Model import MLPProjector
        inside = False
        # whether at the index part of the linked entity
        position = False
        position_buf, mention_buf = '', ''
        new_sent = ''
        table2vec = self.get_table2vecs(table_name)
        log_probs = []
        for n in range(len(sent)):
            if sent[n] == '#':
                if position:
                    i = int(split(position_buf, "row"))
                    j = int(split(position_buf, "col"))

                    features = [i, j]
                    features.extend(
                        np.average(list(self.get_embd(w) for w in sent.split(' ') if w not in ['#', ';', ',']), axis=0))
                    vector = [self.get_embd(w) for w in mention_buf.split(' ')] + [features]
                    if self.model1 is None:
                        self.model1 = MLPProjector(in_dim_1=50, in_dim_2=len(features), out_dim=64)
                        self.model1.train()
                    sent_out = self.model1(vector)[0][0]
                    probs = []
                    all_actions = sorted(table2vec.keys())
                    for i, j in all_actions:
                        e_vector = table2vec[(i, j)]
                        if self.model2 is None:
                            self.model2 = MLPProjector(in_dim_1=50, in_dim_2=len(e_vector[-1]), out_dim=64)
                            self.model2.train()
                        e_out = self.model2(e_vector)[0][0]
                        probs.append(torch.dot(sent_out, e_out))

                    probs = torch.stack(probs)
                    probs = F.softmax(probs, dim=-1)
                    action_probs = torch.distributions.categorical.Categorical(probs)
                    action = action_probs.sample()
                    log_prob = action_probs.log_prob(action)
                    log_probs.append(log_prob)

                    new_sent += f'#{mention_buf};{all_actions[action][0]},{all_actions[action][1]}#'

                    # Reset the buffer
                    position_buf = ""
                    mention_buf = ""
                    inside = False
                    position = False
                else:
                    inside = True
            elif sent[n] == ';':
                position = True
            else:
                if position:
                    position_buf += sent[n]
                elif inside:
                    mention_buf += sent[n]
                else:
                    # non-linked words
                    new_sent += sent[n]
        return new_sent, log_probs

    def initialize_buffer(self, table_name, sent, pos_tag, raw_sent):
        # This function first constructs a masked sentence based on the linked entities. Then, it finds the
        # unlinked numbers and creates count and compare feature vectors for each such number. It reaches a
        # decision based on these vectors to create either a COUNT (count fn) entity or a COMPUTE (avg, sum, etc. fn)
        # entity. For a COMPUTE token, it tries to link it with a table header but is inconsistent. It fails to link it
        # in some cases (case not covered in the code).
        count = 0

        t = self.get_table(table_name)
        cols = t.columns

        # mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}
        mapping = get_col_types(t)

        count += 1
        inside = False
        position = False
        masked_sent = ''
        position_buf, mention_buf = '', ''
        mem_num, head_num, mem_str, head_str, mem_date, head_date, nonlinked_num = [[] for _ in range(7)]
        ent_index = 0
        ent2content = {}
        for n in range(len(sent)):
            if sent[n] == '#':
                if position:
                    if position_buf.startswith('0'):
                        idx = int(split(position_buf, "col"))
                        if mapping[idx] == 'num':
                            if cols[idx] not in head_num:
                                head_num.append(cols[idx])
                        elif mapping[idx] == 'str':
                            if cols[idx] not in head_str:
                                head_str.append(cols[idx])
                        elif mapping[idx] == 'date':
                            if cols[idx] not in head_date:
                                head_date.append(cols[idx])
                        else:
                            raise ValueError(f"Unsupported col type: {mapping[idx]}")
                    else:
                        row = int(split(position_buf, "row"))
                        idx = int(split(position_buf, "col"))
                        if idx == -1:
                            pass
                        else:
                            if mapping[idx] == 'num':
                                if mention_buf.isdigit():
                                    mention_buf = int(mention_buf)
                                else:
                                    try:
                                        mention_buf = float(mention_buf)
                                    except Exception:
                                        pass
                                val = (cols[idx], mention_buf)
                                if val not in mem_num:
                                    mem_num.append(val)
                            elif mapping[idx] == 'str':
                                if len(fuzzy_match(t, cols[idx], mention_buf)) == 0:
                                    val = (cols[idx], mention_buf)
                                else:
                                    val = (cols[idx], mention_buf)
                                if val not in mem_str:
                                    mem_str.append(val)
                            elif mapping[idx] == 'date':
                                val = cols[idx], mention_buf
                                mem_date.append(val)
                            else:
                                raise ValueError(f"Unsupported col type: {mapping[idx]}")
                    # Direct matching
                    masked_sent += "<ENTITY{}>".format(ent_index)
                    ent2content["<ENTITY{}>".format(ent_index)] = str(mention_buf)
                    ent_index += 1
                    # Reset the buffer
                    position_buf = ""
                    mention_buf = ""
                    inside = False
                    position = False
                else:
                    inside = True
            elif sent[n] == ';':
                position = True
            else:
                if position:
                    position_buf += sent[n]
                elif inside:
                    mention_buf += sent[n]
                else:
                    masked_sent += sent[n]

        tokens = masked_sent.split()
        new_tokens = []
        for i in range(len(tokens)):
            _ = tokens[i]
            if _.isdigit():
                num = int(_)
            elif '.' in tokens[i]:
                try:
                    num = float(_)
                except Exception:
                    new_tokens.append(_)
                    continue
            else:
                pat = r'\d(th|nd|rd)'
                _ = _.replace('first', '1st').replace('second', '2nd').replace('third', '3rd').replace(
                    'fourth', '4th').replace('fifth', '5th').replace('sixth', '6th').replace('seventh', '7th').replace(
                    'eighth', '8th').replace('ninth', '9th').replace('tenth', '10th').replace(
                    'eleventh', '11th').replace('twelfth', '12th').replace('thirteenth', '13th').replace(
                    'fourteenth', '14th').replace('fifteenth', '15th')
                if len(re.findall(pat, _)) > 0:
                    reres = re.findall(r'(\d+)(th|nd|rd)', _)
                    if len(reres) == 0:
                        new_tokens.append(_)
                        continue
                    # first number in the first matched group
                    num = reres[0][0]
                    new_tokens.append("<NARG{}>".format(ent_index))
                    ent2content["<NARG{}>".format(ent_index)] = _
                    ent_index += 1
                    mem_num.append(("ntharg", num))
                else:
                    new_tokens.append(_)
                continue

            # Reason whether the number is a comparison or a count
            compare_features = []
            count_features = []

            if tokens[i - 1] in months_b + months_a:
                compare_features.append(-6)
                count_features.append(-6)
            else:
                compare_features.append(0)
                count_features.append(0)

            if any([_ in tokens for _ in ["than", "over", "more", "less", "below", "above", 'least']]):
                compare_features.append(2)
            else:
                compare_features.append(-2)

            if any([_ in pos_tag for _ in ["RBR", "JJR"]]):
                compare_features.append(2)
            else:
                compare_features.append(-2)

            if isinstance(num, int):
                count_features.append(0)
                compare_features.append(0)
            else:
                count_features.append(-6)
                compare_features.append(4)

            if num > 50:
                count_features.append(-4)
                if num > 1900 and num < 2020:
                    compare_features.append(-2)
                else:
                    compare_features.append(2)
            else:
                if num > len(t):
                    compare_features.append(2)
                    count_features.append(-4)
                else:
                    compare_features.append(-1)
                    count_features.append(2)

            if i > 1 and tokens[i - 1] in ['be', 'are', 'is'] and tokens[i - 2] == 'there':
                count_features.append(4)
                compare_features.append(-4)
            elif i + 1 < len(tokens) and tokens[i + 1] == 'of':
                count_features.append(4)
                compare_features.append(-4)
            else:
                count_features.append(-1)
                compare_features.append(1)

            # if len(head_num) > 0:
            #    features.append(1)
            # else:
            #    features.append(0)
            if any([_ not in map(lambda x: x[0], mem_num) for _ in head_num]):
                compare_features.append(2)
                count_features.append(0)

            # Finally reaching a conclusion
            compare_score = sum(compare_features)
            count_score = sum(count_features)

            if compare_score >= count_score:
                if compare_score > 0:
                    if head_num:
                        # Bug: If compare_score > 0 and all hdrs in head_num are referenced by some entity in mem_num,
                        # Then, the current token/word will be masked in the sentence but it won't have a corresponding
                        # mem_num value
                        flag = False
                        for h in head_num:
                            if any([_[0] == h for _ in mem_num]):
                                continue
                            else:
                                mem_num.append((h, num))
                                flag = True
                        if flag:
                            new_tokens.append("<COMPUTE{}>".format(ent_index))
                            ent2content["<COMPUTE{}>".format(ent_index)] = _
                            ent_index += 1
                        else:
                            new_tokens.append("<NONLINKED{}>".format(ent_index))
                            ent2content["<NONLINKED{}>".format(ent_index)] = _
                            ent_index += 1
                            nonlinked_num.append(("nonlink_num", num))
                    else:
                        # Bug: If compare_score > 0 and head_num is empty but also all columns are of type str even
                        # though they may contain some integers. this num token will be masked in the sentence but it
                        # won't have a corresponding mem_num value
                        flag = False
                        for col_idx, k in zip(range(len(cols) - 1, -1, -1), cols[::-1]):
                            if mapping[col_idx] == 'num':
                                mem_num.append((k, num))
                                head_num.append(k)
                                flag = True
                                break
                        if flag:
                            new_tokens.append("<COMPUTE{}>".format(ent_index))
                            ent2content["<COMPUTE{}>".format(ent_index)] = _
                            ent_index += 1
                        else:
                            new_tokens.append("<NONLINKED{}>".format(ent_index))
                            ent2content["<NONLINKED{}>".format(ent_index)] = _
                            ent_index += 1
                            nonlinked_num.append(("nonlink_num", num))
                else:
                    new_tokens.append("<NONLINKED{}>".format(ent_index))
                    ent2content["<NONLINKED{}>".format(ent_index)] = _
                    ent_index += 1
                    nonlinked_num.append(("nonlink_num", num))
                    # new_tokens.append(_)
                    continue
            else:
                if count_score > 0:
                    new_tokens.append("<COUNT{}>".format(ent_index))
                    ent2content["<COUNT{}>".format(ent_index)] = _
                    ent_index += 1
                    mem_num.append(("tmp_input", num))
                else:
                    new_tokens.append("<NONLINKED{}>".format(ent_index))
                    ent2content["<NONLINKED{}>".format(ent_index)] = _
                    ent_index += 1
                    nonlinked_num.append(("nonlink_num", num))
                    # new_tokens.append(_)
                    continue

                    # Correct some wrongly linked count
        to_delete = []
        for index, (k, v) in enumerate(mem_num):
            if k != 'tmp_input' and isinstance(v, int) and v < 10:
                if 'there be {} '.format(v) in raw_sent or 'there are {} '.format(v) in raw_sent or \
                        'there is {} '.format(v) in raw_sent or ' {} of '.format(v) in raw_sent or raw_sent.startswith(
                    str(v) + " "):
                    mem_num[index] = ('tmp_input', v)
                    for k, content in ent2content.items():
                        if content == str(v):
                            new_k = re.sub(r'<[^0-9]+([0-9]+)>', r'<COUNT\1>', k)
                            ent2content[new_k] = v
                            new_tokens[new_tokens.index(k)] = new_k
                            to_delete.append(k)
                            break
                    for k in to_delete:
                        del ent2content[k]
                    to_delete = []

        for k, v in mem_num:
            if k not in head_num and k not in ["tmp_input", 'ntharg']:
                head_num.append(k)

        for k, v in mem_str:
            if k not in head_str:
                head_str.append(k)

        if any([_ in pos_tag for _ in ['RBR', 'RBS', 'JJR', 'JJS']]) and len(head_num) == 0:
            for col_idx, k in zip(range(len(cols) - 1, -1, -1), cols[::-1]):
                if mapping[col_idx] == 'num':
                    head_num.append(k)
                    break

        return (" ".join(new_tokens), mem_str, mem_num, mem_date,
                head_str, head_num, head_date, nonlinked_num, ent2content)

    def run(self, table_name, sent, masked_sent, pos_tag, mem_str, mem_num,
            mem_date, head_str, head_num, head_date, masked_val):
        t = pandas.read_csv(os.path.join(self.folder, table_name), delimiter="#", encoding='utf-8')
        t.fillna('')
        res = dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str,
                                  mem_num, mem_date, head_str, head_num, head_date, masked_val, 7)
        return res[-1]

    def normalize(self, sent):
        sent = sent.lower()
        sent = sent.replace(',', '')
        sent = replace_number(sent)
        sent, _, pos_tags = self.get_lemmatize(sent, True)
        return sent, pos_tags

    def get_aux_objs_mhle(self, mem_num, mem_str, non_linked_num):
        all_entities = mem_num + mem_str + non_linked_num

        def check_eq(a):
            for k, v in all_entities:
                if k == 'ntharg':
                    continue
                if obj_compare(a, v, round=True, type="eq"):
                    return k, v
                if type(v) in [int, float] and obj_compare(a, -v, round=True, type="eq"):
                    return k, v
            return None

        def get_new_header(ent):
            if ent is None:
                return ent
            k, v = ent
            if k == 'tmp_input':
                k = 'msk_input'
            else:
                k = 'msk_' + k
            return k, v

        return all_entities, check_eq, get_new_header

    def mask_highest_lo_entity_1(self, mem_num, mem_str, non_linked_num, logic_json):
        if not isinstance(logic_json, dict):
            return None

        all_entities, check_eq, get_new_header = self.get_aux_objs_mhle(mem_num, mem_str, non_linked_num)

        handled_majority_types = {'most_str_eq', 'all_str_eq', 'most_greater', 'most_less', 'most_greater_eq',
                                  'most_less_eq', 'all_greater', 'all_greater_eq', 'all_less', 'all_less_eq', 'most_eq',
                                  'all_eq'}

        if logic_json['func'] in ['greater', 'less']:
            order = [0, 1] if np.random.uniform() < 0.5 else [1, 0]
            for ix in order:
                arg = logic_json['args'][ix]
                # arg is a hop { filter_eq {
                r = check_eq(arg['args'][0]['args'][-1])
                if r is not None:
                    return get_new_header(r)
        elif logic_json['func'] in handled_majority_types:
            # last arg is the entity
            r = check_eq(logic_json['args'][-1])
            if r is not None:
                return get_new_header(r)
        else:
            for arg in logic_json['args']:
                r = self.mask_highest_lo_entity_1(mem_num, mem_str, non_linked_num, arg)
                if r is not None:
                    return r
        return None

    def mask_highest_lo_entity_2(self, mem_num, mem_str, non_linked_num, logic_json):
        all_entities, check_eq, get_new_header = self.get_aux_objs_mhle(mem_num, mem_str, non_linked_num)

        if logic_json['func'] in ['eq', 'str_eq', 'round_eq']:
            args = logic_json['args']
            if all(type(a) is dict for a in args):
                for a in args:
                    r = self.mask_highest_lo_entity_2(mem_num, mem_str, non_linked_num, a)
                    if r is not None:
                        return r
            else:
                for a in args:
                    if type(a) is dict:
                        continue
                    r = check_eq(a)
                    if r is not None:
                        return get_new_header(r)
        else:
            args = logic_json['args']
            if all(type(a) is dict for a in args):
                for a in args:
                    r = self.mask_highest_lo_entity_2(mem_num, mem_str, non_linked_num, a)
                    if r is not None:
                        return r
        return None

    def parse(self, table_name, og_sent, logic_json, action, debug=False):
        sent, pos_tags = self.normalize(og_sent)
        raw_sent = " ".join(sent)
        linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
        # mem_str, mem_num, mem_date, head_str, head_num, head_date,
        ret_val = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)
        # masked_sent, mem_str, mem_num, mem_date, head_str, head_num, head_date, non_linked_num, mapping = ret_val
        masked_sent, mem_str, mem_num, mem_date, _, _, _, non_linked_num, mapping = ret_val
        head_num, head_str, head_date = self.get_headers(table_name)

        # memory_num and memory_str are (hdr, val) tuples which serve as arguments for the programs of type num and str
        # resp. Whereas, header_num and header_str are useful for arguments of type header_num and header_str resp.
        # if len(mem_num) + len(non_linked_num) == 0:
        #     return None
        # mem_num = self.mask_random_number(mem_num, non_linked_num, mask_num_ix)
        masked_val = None
        if action in ['comparative', 'majority']:
            masked_val = self.mask_highest_lo_entity_1(mem_num, mem_str, non_linked_num, logic_json)
        if masked_val is None:
            masked_val = self.mask_highest_lo_entity_2(mem_num, mem_str, non_linked_num, logic_json)
        if masked_val is None:
            return

        def get_old_val(_x):
            return 'tmp_input' if _x == 'msk_input' else _x[4:]

        def remove_h(_p, _mem):
            new_mem = []
            for __p in _mem:
                if _p == __p:
                    continue
                new_mem.append(__p)
            return new_mem

        og_val = (get_old_val(masked_val[0]), masked_val[1])
        mem_str, mem_num, mem_date = (
            remove_h(og_val, mem_str), remove_h(og_val, mem_num), remove_h(og_val, mem_date))

        if debug:
            print(f"\nInput to dynmiac programmer:\nog_sent: {og_sent}"
                  f"\nlinked: {linked_sent}\nmasked: {masked_sent}\nmapping: {mapping}"
                  f"\nmem_str, mem_num, mem_date: {mem_str, mem_num, mem_date}"
                  f"\nhead_str, head_num, head_date: {head_str, head_num, head_date}"
                  f"\nmasked_val: {masked_val}\n")

        result = self.run(table_name, raw_sent, masked_sent, pos, mem_str, mem_num,
                          mem_date, head_str, head_num, head_date, masked_val)

        c = list(set(result))
        result = [x for x in c if '/True' in x]

        return len(c), result, masked_sent, mapping

    def train_entity_linker(self, data=None):
        import torch
        from torch import optim
        if data is None:
            with open('data/l2t/train_el.json', 'r') as f:
                data = json.load(f)

        def get_old_val(_x):
            return 'tmp_input' if _x == 'msk_input' else _x[4:]

        def remove_h(_p, _mem):
            new_mem = []
            for __p in _mem:
                if _p == __p:
                    continue
                new_mem.append(__p)
            return new_mem

        total_covered = 0
        for table_name, og_sent, logic_json, action in tqdm(data):
            sent, pos_tags = self.normalize(og_sent)
            raw_sent = " ".join(sent)
            linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
            sent, log_probs = self.new_entity_link(table_name, linked_sent, pos)
            ret_val = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)
            masked_sent, mem_str, mem_num, mem_date, head_str, head_num, head_date, non_linked_num, mapping = ret_val

            masked_val = None
            if action in ['comparative', 'majority']:
                masked_val = self.mask_highest_lo_entity_1(mem_num, mem_str, non_linked_num, logic_json)
            if masked_val is None:
                masked_val = self.mask_highest_lo_entity_2(mem_num, mem_str, non_linked_num, logic_json)
            if masked_val is None:
                continue

            og_val = (get_old_val(masked_val[0]), masked_val[1])
            mem_str, mem_num, mem_date = (
                remove_h(og_val, mem_str), remove_h(og_val, mem_num), remove_h(og_val, mem_date))
            result = self.run(table_name, raw_sent, masked_sent, pos, mem_str, mem_num,
                              mem_date, head_str, head_num, head_date, masked_val)

            c = list(set(result))
            result = [x for x in c if '/True' in x]
            rewards = [0] * len(log_probs)
            if len(result) > 0:
                rewards[-1] = 1
                total_covered += 1

            if self.optimizer is None:
                self.optimizer = optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=1e-4)
            R = 0
            policy_loss = []
            returns = []
            eps = np.finfo(np.float32).eps.item()
            for r in rewards[::-1]:
                R = r + 1 * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + eps)
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)

            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
        print(f"Total covered: {total_covered}")
        return

    def hash_string(self, string):
        import hashlib
        sha = hashlib.sha256()
        sha.update(string.encode())
        return sha.hexdigest()[:16]

    def get_headers(self, table_name):
        t = self.get_table(table_name)
        mapping = get_col_types(t)
        head_str, head_num, head_date = [[] for _ in range(3)]
        for i, col in enumerate(t.columns):
            if mapping[i] == 'num':
                head_num.append(col)
            elif mapping[i] == 'date':
                head_date.append(col)
            elif mapping[i] == 'str':
                head_str.append(col)
        return head_num, head_str, head_date

    def distribute_parse(self, inputs):
        table_name, sent, logic_json, action = inputs
        # Get filename for caching
        formatted_sent = re.sub(r"[^\w\s]", '', sent)
        formatted_sent = re.sub(r"\s+", '-', formatted_sent)
        if len(formatted_sent) > 100:
            formatted_sent = formatted_sent[-100:]
        formatted_sent = table_name + '-' + formatted_sent
        if not os.path.exists('tmp/results/{}.json'.format(formatted_sent)):
            sent, pos_tags = self.normalize(sent)
            raw_sent = " ".join(sent)
            try:
                linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
            except AssertionError:
                return None
            # mem_str, mem_num, mem_date, head_str, head_num, head_date,
            ret_val = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)
            masked_sent, mem_str, mem_num, mem_date, _, _, _, non_linked_num, mapping = ret_val
            head_num, head_str, head_date = self.get_headers(table_name)

            masked_val = None
            if action in ['comparative', 'majority']:
                masked_val = self.mask_highest_lo_entity_1(mem_num, mem_str, non_linked_num, logic_json)
            if masked_val is None:
                masked_val = self.mask_highest_lo_entity_2(mem_num, mem_str, non_linked_num, logic_json)
            if masked_val is None:
                return None

            def get_old_val(_x):
                return 'tmp_input' if _x == 'msk_input' else _x[4:]

            def remove_h(_p, _mem):
                new_mem = []
                for __p in _mem:
                    if _p == __p:
                        continue
                    new_mem.append(__p)
                return new_mem

            og_val = (get_old_val(masked_val[0]), masked_val[1])
            mem_str, mem_num, mem_date = (
                remove_h(og_val, mem_str), remove_h(og_val, mem_num), remove_h(og_val, mem_date))

            result = self.run(table_name, raw_sent, masked_sent, pos, mem_str, mem_num,
                              mem_date, head_str, head_num, head_date, masked_val)
            c = list(set(result))
            result = [x for x in c if '/True' in x]

            ret_val = inputs[0], inputs[1], linked_sent, masked_val, mem_str, mem_num, mem_date, len(c), result

            with open('tmp/results/{}.json'.format(formatted_sent), 'w') as f:
                json.dump(ret_val, f, indent=2)

            return ret_val
        else:
            with open('tmp/results/{}.json'.format(formatted_sent), 'r') as f:
                data = json.load(f)
            return data


def test_1():
    def get_logic_json(tbl, sent):
        with open('data/l2t/all_data.json') as f:
            data = json.load(f)
        for ent in data:
            if (ent['url'] ==
                    'https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/master/data/all_csv/' + tbl):
                if ent['sent'] == sent:
                    return ent['logic'], ent['action']
        raise ValueError("Bad input")

    parser = Parser("data/l2t/all_csv")

    def parse_it(tbl, sent, alt_sent=None):
        if alt_sent is None:
            alt_sent = sent

        print("Title:", parser.title_mapping[tbl])
        print("Table:", tbl)
        a, b = get_logic_json(tbl, sent)
        return parser.parse(tbl, alt_sent, a, b, True)

    # parse_it("2-14173105-18.html.csv",
    #          "in the 1999-2000 philadelphia flyers season , the player who was selected 2nd is jeff feniak .")

    # Works
    # print(parse_it("2-13014020-6.html.csv",
    #                "philipp petzschner partnered with j\u00fcrgen melzer "
    #                "for the majority of his tennis doubles tournaments ."))
    # print(parse_it("2-17443121-2.html.csv",
    #                "the episode entitled gary marries off his ex aired seven days after gary gets boundaries ."))
    # print(parse_it("1-1342233-3.html.csv",
    #                "in the election of 1946 for united states house of representatives ,"
    #                " the incumbent who was first elected the second earliest was frank w boykin ."))
    # print(parse_it("2-15740721-1.html.csv",
    #                "1994 was the second year that juraj brestovsk\u00fd won "
    #                "the men 's singles in the slovak national badminton championships ."))
    # print(parse_it("2-14202514-1.html.csv",
    #                "the miami dolphins ' match against detroit lions was the earliest of the 1972 season games ."))

    # Warning about regex
    # parse_it("2-12326046-2.html.csv",
    #          "the opole tournament was the only one in which ana jovanovi\u0107 used a carpet ( i ) surface .")

    # print(parse_it("2-11963536-8.html.csv",
    #                "the match on 16 march 2008 had the highest attendance of all the matches ."))

    # To try out
    # Something is wrong in this one. Debug this
    print(parse_it("1-24538587-11.html.csv",
                   "the core i3 - 32xx brand name processor has the second highest tdp of intel core processors ."))

    # No programs
    ## reason: Entity linker is not able to link the word "resignation" to the entity "resigned march 4 , 1894"
    # print(parse_it("1-2417445-4.html.csv",
    #                "in the 54th united states congress , of the successors that took office in 1895 ,"
    #                " the only time that the vacancy was due to a resignation was when the vacator was "
    #                "edwin j jordan ."))
    ## reaason: Avg position is 95 and round=True in comparison is not able to make 100 equal to 95
    # print(parse_it("2-1027162-1.html.csv",
    #                "the lauryn williams competitions have an aggregate position of about 100 m."))
    ## reason: No trigger words for sum present here
    # print(parse_it("1-27922491-8.html.csv",
    #                "the members of the somerset county cricket club in 2009 played in 84 matches ."))
    # We need the "attendance" column as a header but "attended" word is never linked to that header
    # print(parse_it("1-10361453-2.html.csv",
    #                "the game against green bay packers on september 15 was "
    #                "better attended than the game against chicago bears on november 3 ."))
    ## reason: ('episode', 'gary get boundaries') is the mem_str we get, but the actual entity is 'gary gets boundaries'
    # we are not able to do a filter
    # print(parse_it("2-17443121-2.html.csv",
    #                "the episode entitled gary marries off his ex aired seven days after gary gets boundaries ."))
    ## reason: 10 is thought of as a number by the entity linker
    # print(parse_it("1-1341423-35.html.csv",
    #                "for the united states house of representatives election in 2000 ,"
    #                " of the incumbents that were re-elected , the one with the 2nd most "
    #                "recent first election date was from ohio district 10 ."))
    ## reason: 11 years is not mapped to the column 'from' where column from contains years
    # print(parse_it("2-11482079-2.html.csv",
    #                "alvan adams began playing for the phoenix suns 11 years before rafael addison ."))

    # Wrong program
    ## reason: 15 days is not linked to the column "date"
    # print(parse_it("2-1140080-2.html.csv",
    #                "in the 1979 formula one season , "
    #                "the german grand prix was 15 days after the british grand prix ."))
    ## reason: april is masked and all_eq_inv doesn't return max common substr. Moreover, col date is not of date type
    ## and therefore, all_eq_inv (num type) will be called and all dates are not equal. Just the month is equal.
    # print(parse_it("1-17311759-9.html.csv",
    #                "all games of the atlanta hawks ' in the 2008 - 09 season were played in the month of april ."))

    # No masking
    ## reason: 1970s was not linked to the column date of birth (of format 07.09.1970) and it was the only entity
    # print(parse_it("2-15859432-3.html.csv",
    #                "most of the men 's volleyball players at the 2004 summer olympics were born in to 1970s ."))
    ## reason: The masked value should be "won" as there is a column named "result" in the table with values like
    ## 'won 4 - 0'. The generated program would be most_eq_inv { all_rows ; result } = won/True
    # print(parse_it("2-15213262-10.html.csv",
    #                "the flames won most of their games during the 07-08 season ."))
    ## reason: George h. Christopher was transformed to george christopher before entity linking. Even though, it was
    ## linked correctly, we were not able to see that the linked entity "george christopher" appears in the logic form
    ## eq { hop { nth_argmin { all_rows ; reason for change ; 1 } ; vacator } ; george h christopher ( d ) } = true
    # print(parse_it("1-2159571-2.html.csv",
    #                "the first us representative in the 86th us congress to be replaced was george h. christopher ."))
    ## reason: All entities are linked but there is no mention of entities like new york and beneath vesuvius
    ## in the sentence.
    ## less { hop { filter_eq { all_rows ; episode title ; new york } ; original airdate } ;
    ## hop { filter_eq { all_rows ; episode title ; beneath vesuvius } ; original airdate } } = true
    # print(parse_it("2-10926568-1.html.csv",
    #                "in june 2007 , eric geller hosted an episode of \" cities "
    #                "of the underworld \" before don wildman did ."))
    return


def test_2():
    with open('data/l2t/train.json') as f:
        data = json.load(f)
    entry = data[np.random.choice(len(data), 1)[0]]
    sent = entry['sent']
    table = entry['url'][entry['url'].find('all_csv/') + 8:]
    logic_json = entry['logic']
    action = entry['action']

    parser = Parser("data/l2t/all_csv")
    check = parser.parse(table, sent, logic_json, action, True)
    while check is None:
        entry = data[np.random.choice(len(data), 1)[0]]
        sent = entry['sent']
        table = entry['url'][entry['url'].find('all_csv/') + 8:]
        logic_json = entry['logic']
        action = entry['action']
        print(entry['logic_str'])
        check = parser.parse(table, sent, logic_json, action, True)

    print("Table:", table)
    print("Title:", parser.title_mapping[table])
    print("Sent:", sent)
    print(check)
    return


def test_3(idx=None):
    parser = Parser("data/all_csv")

    with open('data/l2t/train.json', 'r') as f:
        data = json.load(f)

    if idx is not None:
        data = [data[idx]]

    for entry in data:
        sent = entry['sent']
        table_name = entry['url'][entry['url'].find('all_csv/') + 8:]
        logic_json = entry['logic']
        action = entry['action']
        print("Table:", table_name)
        print("Title:", parser.title_mapping[table_name])
        print("Sent:", sent)
        print(parser.parse(table_name, sent, logic_json, action, True))

        print()
    return


def generate_programs():
    parser = Parser("data/all_csv")

    with open('data/l2t/train.json', 'r') as f:
        data = json.load(f)

    args = []
    for entry in data:
        sent = entry['sent']
        table_name = entry['url'][entry['url'].find('all_csv/') + 8:]
        logic_json = entry['logic']
        action = entry['action']
        args.append((table_name, sent, logic_json, action))

    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(parser.distribute_parse, args, chunksize=1), total=len(args)))
        results = list(tqdm(pool.map(parser.distribute_parse, args), total=len(args)))

    with open("data/programs.json", 'w') as f:
        json.dump(results, f, indent=2)
    return


def sample_unmasked_sents():
    with open('data/programs.json', 'r') as f:
        data = json.load(f)

    def get_metadata():
        with open('data/l2t/train.json') as f:
            data = json.load(f)
        ret_dict = {}
        ret_dict_2 = {}
        for ent in tqdm(data):
            tname = ent['url'][ent['url'].find('all_csv/') + 8:]
            ret_dict[ent['sent']] = tname, ent['logic'], ent['action']
            ret_dict_2[ent['sent']] = 0
        return ret_dict, ret_dict_2

    sent2logic, sent2covered = get_metadata()

    for entry in tqdm(data):
        if entry is None:
            continue
        sent = entry[1]
        sent2covered[sent] += 1
    print("Multiple sents:", len([v for v in sent2covered.values() if v > 1]))
    print([k for k, v in sent2covered.items() if v > 1])

    all_uncovered_sents = []
    for sent in tqdm(sent2covered):
        if sent2covered[sent] > 0:
            continue
        table_name, logic_json, action = sent2logic[sent]
        all_uncovered_sents.append((table_name, sent, logic_json, action))

    sampled_sents_idx = np.random.choice(len(all_uncovered_sents), 60)
    sampled_sents = [all_uncovered_sents[i] for i in sampled_sents_idx]
    with open('data/l2t/sampled_unmasked_sents.json', 'w') as f:
        json.dump(sampled_sents, f, indent=2)
    return


def create_train_data():
    with open('data/programs.json', 'r') as f:
        data = json.load(f)

    def get_metadata():
        with open('data/l2t/train.json') as f:
            data = json.load(f)
        ret_dict = {}
        for ent in tqdm(data):
            ret_dict[ent['sent']] = ent['logic'], ent['action']
        return ret_dict

    sent2logic = get_metadata()

    all_uncovered_sents = []
    for entry in tqdm(data):
        if entry is None or len(entry[-1]) > 0:
            continue
        table_name, sent = entry[0], entry[1]
        logic_json, action = sent2logic[sent]
        all_uncovered_sents.append((table_name, sent, logic_json, action))

    sampled_sents_idx = np.random.choice(len(all_uncovered_sents), 60)
    sampled_sents = [all_uncovered_sents[i] for i in sampled_sents_idx]
    with open('data/l2t/train_el.json', 'w') as f:
        json.dump(sampled_sents, f, indent=2)
    return


def train_el(num_episodes):
    with open('data/l2t/train_el.json', 'r') as f:
        data = json.load(f)
    parser = Parser("data/all_csv")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for i in range(num_episodes):
            random.shuffle(data)
            parser.train_entity_linker(data)
    return


if __name__ == "__main__":
    # test_1()
    # test_3(6127)
    generate_programs()
    # sample_unmasked_sents()
    # create_train_data()
    # train_el(1000)
