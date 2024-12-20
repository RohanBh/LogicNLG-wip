import json
import pprint
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from utils import plotCDF


def print_succ_masks(data):
    tot_msks = len([x for x in data if x is not None])
    print(f"Total statements: {len(data)}, masked: {tot_msks}, unmasked: {len(data) - tot_msks}")
    return


def print_coverage(data):
    tot_stmts = len([x for x in data if x is not None])
    total_progs = 0
    for entry in data:
        if entry is None:
            continue
        progs = entry[-1]
        if len(progs) > 0:
            total_progs += 1
    print(f'Masked stmts: {tot_stmts}, Progs found: {total_progs}, Coverage: {total_progs * 100 / tot_stmts}')
    return


def show_dist(data):
    counter = {}
    for entry in data:
        if entry is None:
            continue
        progs = entry[-1]
        for prog in progs:
            prog_type = prog.split('{')[0].strip()
            if prog_type == 'hop':
                prog_type = prog.split('{')[1].strip()
            if prog_type not in counter:
                counter[prog_type] = 0
            counter[prog_type] += 1
    print("Program types:")
    new_counter = dict(sorted(counter.items(), key=lambda x: (-x[-1])))
    pprint.pprint(new_counter, indent=2, sort_dicts=False)
    return


def num_progs_cdf(data, is_log=False):
    tot_progs_arr = []
    for entry in data:
        if entry is None:
            continue
        progs = entry[-1]
        if len(progs) > 200:
            print(entry[:2])
        tot_progs_arr.append(len(progs))
    ax = plt.gca()
    plotCDF(ax, {0: tot_progs_arr}, {0: ''}, 'Total Programs', '# of instances', isLog=is_log)
    locs = plt.yticks()[0]
    plt.yticks(locs, [f'{float(f) * len(tot_progs_arr):.0f}' for f in locs])
    print("Max programs:", sorted(tot_progs_arr)[-10:])
    print_dict = dict(sorted(Counter(tot_progs_arr).items(), key=lambda x: (x[-1], x[0]))[-10:])
    print(print_dict)
    plt.show()
    return


def main():
    with open("data/programs.json", 'r') as f:
        data = json.load(f)
    print_succ_masks(data)
    print_coverage(data)
    show_dist(data)
    num_progs_cdf(data, is_log=True)
    print("#" * 100)
    with open("data/programs_filtered.json", 'r') as f:
        data = json.load(f)
    print_succ_masks(data)
    print_coverage(data)
    show_dist(data)
    num_progs_cdf(data)
    return


# Amalyze model outputs.
def print_check_proportion(path):
    with open(path, 'r') as f:
        data = json.load(f)
    data = [(x[-1], x[-2] != None) for x in data]
    ctr = Counter(data)
    ctr2 = {
        'FN': ctr[(False, False)],
        'FP': ctr[(False, True)],
        'TP': ctr[(True, True)],
    }
    print(ctr2)
    assert len(data) == sum(ctr2.values())
    print("Coverage (TP / (TP + FP + FN)):", ctr2['TP'] / len(data))
    print("Recall (TP / (TP + FN)):", ctr2['TP'] / (ctr2['TP'] + ctr2['FN']))
    print("Precision (TP / (TP + FP)):", ctr2['TP'] / (ctr2['TP'] + ctr2['FP']))
    return


def show_dist_2(path):
    with open(path, 'r') as f:
        data = json.load(f)
    counter = {'none': 0}
    tot_prog_dict = {'none': 0}
    for entry in data:
        prog = entry[-2]
        is_accept = entry[-1]
        if prog is None:
            counter['none'] += 1
            tot_prog_dict['none'] += 1
            continue
        prog_type = prog.split('{')[0].strip()
        if prog_type == 'hop':
            prog_type = prog.split('{')[1].strip()
            if '=' in prog_type and '/False' in prog_type:
                prog_type = 'hop'
        if prog_type not in counter:
            counter[prog_type] = 0
        if prog_type not in tot_prog_dict:
            tot_prog_dict[prog_type] = 0
        if is_accept:
            counter[prog_type] += 1
        tot_prog_dict[prog_type] += 1

    counter = {k: (v, tot_prog_dict[k]) for k, v in counter.items()}
    new_counter = dict(sorted(counter.items(), key=lambda x: x[-1][1], reverse=True))
    pprint.pprint(new_counter, indent=2, sort_dicts=False)
    return


# Use this with Ranker output
def show_dist_3(path):
    with open(path, 'r') as f:
        data = json.load(f)
    counter = {'none': 0}
    tot_prog_dict = {'none': 0}
    for entry in data:
        prog = [x for x in entry[-2] if '/True' in x]
        if len(prog) == 0:
            prog = entry[-2][0]
        else:
            prog = prog[0]
        is_accept = entry[-1]
        if prog is None:
            counter['none'] += 1
            tot_prog_dict['none'] += 1
            continue
        prog_type = prog.split('{')[0].strip()
        if prog_type == 'hop':
            prog_type = prog.split('{')[1].strip()
            if '=' in prog_type and '/False' in prog_type:
                prog_type = 'hop'
        if prog_type not in counter:
            counter[prog_type] = 0
        if prog_type not in tot_prog_dict:
            tot_prog_dict[prog_type] = 0
        if is_accept:
            counter[prog_type] += 1
        tot_prog_dict[prog_type] += 1

    counter = {k: (v, tot_prog_dict[k]) for k, v in counter.items()}
    new_counter = dict(sorted(counter.items(), key=lambda x: x[-1][1], reverse=True))
    pprint.pprint(new_counter, indent=2, sort_dicts=False)
    return


def main2():
    # path = Path("plstm_outputs/out_rbn_129.json")
    path = Path("plstm_outputs/out_plstm_with_beam_search_and_ranker__10_top_k_5.json")
    # path = Path("roberta_ranker_outputs/out_004.json")
    # TODO: Fix this. Prints wrong metrics as roberta outputs discard FNs.
    print_check_proportion(path)
    show_dist_2(path)
    # show_dist_3(path)
    return

def analyze_out_34():
    with open('gpt2_lm_outputs/out_034.json') as fp:
        data = json.load(fp)
    null_stmts = 0
    counter = {}
    for k, v in data.items():
        for _v in v:
            if _v[1] is None:
                null_stmts += 1
                continue
            prog = _v[1]
            prog_type = prog.split('{')[0].strip()
            if prog_type == 'hop':
                prog_type = prog.split('{')[1].strip()
                if '=' in prog_type and '/False' in prog_type:
                    prog_type = 'hop'
            if prog_type not in counter:
                counter[prog_type] = 0
            counter[prog_type] += 1
    pprint.pprint(counter)
    return

if __name__ == '__main__':
    # main()
    main2()
    # analyze_out_34()
