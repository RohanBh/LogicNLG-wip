import json
import pprint
import re
from pathlib import Path

import numpy as np


def create_csvs():
    base_dir = Path('data', 'l2t', 'all_csv')
    base_dir.mkdir(parents=True, exist_ok=True)

    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    finished = set()
    for entry in data:
        i = entry['url'].find('all_csv/')
        name = entry['url'][i + 8:]
        if name in finished:
            continue
        with open(base_dir / name, 'w') as fp:
            fp.write('#'.join(entry['table_header']))
            fp.write('\n')
            for row in entry['table_cont']:
                fp.write('#'.join(row))
                fp.write('\n')
        finished.add(name)
    return


def print_all_first_funcs():
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    counter = {}
    for entry in data:
        fn = entry['logic']['func']
        if fn not in counter:
            counter[fn] = 0
        counter[fn] += 1

    total = len(data)
    for k in counter.keys():
        counter[k] = round(100 * counter[k] / total, 2)

    # new_counter = {}
    for k, v in sorted(list(counter.items()), key=lambda x: x[1], reverse=True):
        print(k, ':', v)
        # new_counter[k] = v

    # import pprint
    # pprint.pprint(new_counter)
    return


def print_type_2_func():
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    def recursive_get(logic_json):
        all_funcs = {logic_json['func']}
        for arg in logic_json['args']:
            if type(arg) is dict:
                all_funcs.update(recursive_get(arg))
            else:
                continue
        return all_funcs

    type2set = {}
    for entry in data:
        action = entry['action']
        if action not in type2set:
            type2set[action] = set()
        type2set[action].update(recursive_get(entry['logic']))

    for action in type2set:
        type2set[action] -= set(f for f in type2set[action] if 'filter' in f or 'hop' in f or 'and' in f)
    pprint.pprint(type2set)
    return


def print_type_2_funccount(sorted_order=False):
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    def update_ctr(d1, d2):
        for k, v in d1.items():
            if k not in d2:
                d2[k] = 0
            d2[k] += v
        return

    def recursive_get(logic_json):
        all_funcs = {logic_json['func']: 1}
        for arg in logic_json['args']:
            if type(arg) is dict:
                update_ctr(recursive_get(arg), all_funcs)
            else:
                continue
        return all_funcs

    type2set = {}
    for entry in data:
        action = entry['action']
        if action not in type2set:
            type2set[action] = {}
        update_ctr(recursive_get(entry['logic']), type2set[action])

    helper_funcs = ['round_eq', 'eq', 'not_eq', 'not_str_eq', 'str_eq']
    for action in type2set:
        for k in list(type2set[action].keys()):
            if 'filter' in k or 'hop' in k or 'and' in k or k in helper_funcs:
                del type2set[action][k]

    if not sorted_order:
        pprint.pprint(type2set)
    else:
        new_type2set = {}
        for action in sorted(type2set.keys()):
            pairs = sorted(type2set[action].items(), key=lambda x: (x[-1], x[0]), reverse=True)
            new_type2set[action] = dict(pairs)
        pprint.pprint(new_type2set, sort_dicts=False)

    # pairs = sorted(type2set['majority'].items(), key=lambda x: (x[-1], x[0]), reverse=True)
    # print("Majority:")
    # pprint.pprint(dict(pairs), sort_dicts=False)
    return


def check_ordinal_property():
    """This function checks if there are some ordinal types which don't have a 1st, 2nd
    type of number in their sentence"""
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    count = 0
    tot_ord_ct = 0
    pat = r'\d(th|nd|rd)'
    fn_sents = []
    for entry in data:
        if entry['action'] != 'ordinal':
            continue
        tot_ord_ct += 1
        # replace tokens
        sent = entry['sent']
        sent = sent.replace('first', '1st').replace('second', '2nd').replace('third', '3rd').replace(
            'fourth', '4th').replace('fifth', '5th').replace('sixth', '6th').replace('seventh', '7th').replace(
            'eighth', '8th').replace('ninth', '9th')
        if len(re.findall(pat, sent)) == 0:
            count += 1
            fn_sents.append(sent)
    print("Total ordinals with no matches", count, 'out of', tot_ord_ct)
    print(np.random.choice(fn_sents, 10))
    return


def check_comp_property():
    """This function checks
    1. if there are some comparative types (less or greater) which do a comparison on non-hops
        Ans - no. All is as expected
    2. if there are some comps which don't have a filter_str_eq inside their hops
    """
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    def get_main_logic(logic_json):
        if type(logic_json) is not dict:
            return None

        if logic_json['func'] in ['greater', 'less']:
            l, r = logic_json['args']
            b1 = 'hop' in l['func'] and 'hop' in r['func']
            if b1:
                b2 = l['args'][0]['func'] == 'filter_str_eq' and r['args'][0]['func'] == 'filter_str_eq'
            else:
                b2 = False
            return logic_json['tostr'], b1, b2

        res = [get_main_logic(arg) for arg in logic_json['args']]
        res = [x for x in res if x is not None]
        if len(res) > 1:
            print("Wow!!!!!")
        return res[0] if len(res) > 0 else None

    count = 0
    tot_ct = 0
    fn_strs = []
    for entry in data:
        if entry['action'] != 'comparative':
            continue
        ret = get_main_logic(entry['logic'])
        if ret is None:
            continue
        tot_ct += 1
        logic_str, ret_val_1, ret_val_2 = ret
        if not ret_val_2:
            fn_strs.append(logic_str)
            count += 1

    print("Total comps with no two hops", count, 'out of', tot_ct)
    if len(fn_strs) > 0:
        print(np.random.choice(fn_strs, 10))
    return


def check_num_rows():
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    row_list = []
    for entry in data:
        row_list.append(len(entry['table_cont']))

    print(max(row_list), min(row_list))
    return


if __name__ == '__main__':
    # print_all_first_funcs()
    # print_type_2_func()
    print_type_2_funccount(sorted_order=True)
    # check_ordinal_property()
    # check_comp_property()
    # check_num_rows()
