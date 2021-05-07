import json
import pprint
from collections import Counter

from utils import plotCDF
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    main()
