import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from utils import powerset

# Outdated
# type2funcs = {'aggregation': {'avg', 'sum'},
#               'comparative': {'diff', 'greater', 'less'},
#               'count': {'count'},
#               'majority': {'all_eq',
#                            'all_greater',
#                            'all_greater_eq',
#                            'all_less',
#                            'all_less_eq',
#                            'all_not_eq',
#                            'all_str_eq',
#                            'most_eq',
#                            'most_greater',
#                            'most_greater_eq',
#                            'most_less',
#                            'most_less_eq',
#                            'most_not_eq',
#                            'most_str_eq',
#                            'most_str_not_eq'},
#               'ordinal': {'nth_argmax', 'nth_min', 'nth_argmin', 'nth_max'},
#               'superlative': {'argmax', 'max', 'argmin', 'min'},
#               'unique': {'only'}
#               }

# Outdated
# trigger_words_data = {'all_eq': {'_count': 41, 'all': 39},
#                       'all_greater': {'_count': 22, 'all': 18},
#                       'all_greater_eq': {'_count': 14, 'all': 13, 'at least': 10},
#                       'all_less': {'_count': 4, 'all': 4, 'under': 2},
#                       'all_less_eq': {'_count': 1},
#                       'all_not_eq': {'_count': 1},
#                       'all_str_eq': {'_count': 238, 'all': 218},
#                       'argmax': {'_count': 790, 'highest': 441, 'most': 182},
#                       'argmin': {'_count': 229, 'earliest': 49, 'highest': 28,
#                                  'lowest': 25, 'all': 25, 'least': 16, 'best': 15},
#                       'avg': {'_count': 811, 'average': 754},
#                       'count': {'_count': 1623, 'total': 214},
#                       'diff': {'_count': 104, 'before': 39},
#                       'greater': {'_count': 476, 'higher': 127},
#                       'less': {'_count': 366, 'earlier': 121, 'before': 116},
#                       'max': {'_count': 111, 'highest': 53},
#                       'min': {'_count': 43, 'earliest': 17},
#                       'most_eq': {'_count': 80, 'most': 45, 'majority': 27},
#                       'most_greater': {'_count': 151, 'most': 81, 'majority': 65},
#                       'most_greater_eq': {'_count': 55, 'least': 41, 'most': 30, 'majority': 24},
#                       'most_less': {'_count': 105, 'most': 69, 'majority': 34, 'under': 29, 'before': 19},
#                       'most_less_eq': {'_count': 7, 'majority': 4, 'most': 3},
#                       'most_not_eq': {'_count': 2, 'most': 2},
#                       'most_str_eq': {'_count': 750, 'most': 377, 'majority': 345},
#                       'most_str_not_eq': {'_count': 8, 'most': 5, 'majority': 3},
#                       'nth_argmax': {'_count': 553, 'highest': 359, '2nd': 165,
#                                      'second': 152, 'most': 59, 'third': 53,
#                                      'among': 50, 'last': 39, '3rd': 30, 'all': 29},
#                       'nth_argmin': {'_count': 365, 'second': 110, 'earliest': 102,
#                                      '2nd': 55, 'lowest': 44, 'third': 37},
#                       'nth_max': {'_count': 29, 'highest': 10, 'second': 8, 'third': 5, 'largest': 5, 'newest': 2},
#                       'nth_min': {'_count': 69, 'second': 10, '1st': 9, '3rd': 5, 'third': 4, 'fourth': 4, '2nd': 4},
#                       'only': {'_count': 1277},
#                       'sum': {'_count': 335, 'total': 284, 'combined': 54}
#                       }

memory_arg_funcs = ('filter_str_eq', 'filter_str_not_eq', 'filter_eq', 'filter_not_eq', 'filter_less',
                    'filter_greater', 'filter_greater_eq', 'filter_less_eq', 'greater_str_inv', 'less_str_inv')

APIs = {}

# With only one argument

APIs['inc'] = {"argument": ['any'], 'output': 'any',
               "function": lambda t: t,
               "tostr": lambda t: "inc {{ {} }}".format(t),
               "tosstr": lambda t: "inc {{ {} }}".format(t),
               'append': False,
               'model_args': ['any']}

APIs['dec'] = {"argument": ['any'], 'output': 'none',
               "function": lambda t: None,
               "tostr": lambda t: "dec {{ {} }}".format(t),
               "tosstr": lambda t: "dec {{ {} }}".format(t),
               'append': False,
               'model_args': ['any']}

### count
APIs['count'] = {"argument": ['row'], 'output': 'num',
                 'function': lambda t: len(t),
                 'tostr': lambda t: "count {{ {} }}".format(t),
                 "tosstr": lambda t: "count {{ {} }}".format(t),
                 'append': True,
                 'model_args': ['row']}

### unique
# APIs['only'] = {"argument": ['row'], 'output': 'bool',
#                 "function": lambda t: len(t) == 1,
#                 "tostr": lambda t: "only {{ {} }}".format(t),
#                 'append': None}

# With only two argument and the first is row
APIs['hop'] = {"argument": ['row', 'header'], 'output': 'obj',
               'function': lambda t, col: hop_op(t, col),
               'tostr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
               'tosstr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
               'append': True,
               'model_args': ['row', 'header_any']}

# APIs['num_hop'] = {"argument": ['row', 'header'], 'output': 'num',
#                    'function': lambda t, col: hop_op(t, col),
#                    'tostr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
#                    'append': True}

APIs['avg'] = {"argument": ['row', 'header_num'], 'output': 'num',
               "function": lambda t, col: agg(t, col, "avg"),
               "tostr": lambda t, col: "avg {{ {} ; {} }}".format(t, col),
               "tosstr": lambda t, col: "avg {{ {} ; {} }}".format(t, col),
               'append': True,
               'model_args': ['row', 'header_num']}

APIs['sum'] = {"argument": ['row', 'header_num'], 'output': 'num',
               "function": lambda t, col: agg(t, col, "sum"),
               "tostr": lambda t, col: "sum {{ {} ; {} }}".format(t, col),
               "tosstr": lambda t, col: "sum {{ {} ; {} }}".format(t, col),
               'append': True,
               'model_args': ['row', 'header_num']}

APIs['max'] = {"argument": ['row', 'header'], 'output': 'obj',
               "function": lambda t, col: nth_maxmin(t, col, order=1, max_or_min="max", arg=False),
               "tostr": lambda t, col: "max {{ {} ; {} }}".format(t, col),
               "tosstr": lambda t, col: "max {{ {} ; {} }}".format(t, col),
               'append': True,
               'model_args': ['row', 'header']}

APIs['min'] = {"argument": ['row', 'header'], 'output': 'obj',
               "function": lambda t, col: nth_maxmin(t, col, order=1, max_or_min="min", arg=False),
               "tostr": lambda t, col: "min {{ {} ; {} }}".format(t, col),
               "tosstr": lambda t, col: "min {{ {} ; {} }}".format(t, col),
               'append': True,
               'model_args': ['row', 'header']}

APIs['argmax'] = {"argument": ['row', 'header'], 'output': 'row',
                  'function': lambda t, col: nth_maxmin(t, col, order=1, max_or_min="max", arg=True),
                  'tostr': lambda t, col: "argmax {{ {} ; {} }}".format(t, col),
                  'tosstr': lambda t, col: "argmax {{ {} ; {} }}".format(t, col),
                  'append': False,
                  'model_args': ['row', 'header']}

APIs['argmin'] = {"argument": ['row', 'header'], 'output': 'row',
                  'function': lambda t, col: nth_maxmin(t, col, order=1, max_or_min="min", arg=True),
                  'tostr': lambda t, col: "argmin {{ {} ; {} }}".format(t, col),
                  'tosstr': lambda t, col: "argmin {{ {} ; {} }}".format(t, col),
                  'append': False,
                  'model_args': ['row', 'header']}

# add for ordinal
APIs['nth_argmax'] = {"argument": ['row', 'header', 'num'], 'output': 'row',
                      'function': lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="max", arg=True),
                      'tostr': lambda t, col, ind: "nth_argmax {{ {} ; {} ; {} }}".format(t, col, ind),
                      'tosstr': lambda t, col, ind: "nth_argmax {{ {} ; {} ; {} }}".format(t, col, ind),
                      'append': False,
                      'model_args': ['row', 'header', 'n']}

APIs['nth_argmin'] = {"argument": ['row', 'header', 'num'], 'output': 'row',
                      'function': lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="min", arg=True),
                      'tostr': lambda t, col, ind: "nth_argmin {{ {} ; {} ; {} }}".format(t, col, ind),
                      'tosstr': lambda t, col, ind: "nth_argmin {{ {} ; {} ; {} }}".format(t, col, ind),
                      'append': False,
                      'model_args': ['row', 'header', 'n']}

APIs['nth_max'] = {"argument": ['row', 'header', 'num'], 'output': 'obj',
                   "function": lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="max", arg=False),
                   "tostr": lambda t, col, ind: "nth_max {{ {} ; {} ; {} }}".format(t, col, ind),
                   "tosstr": lambda t, col, ind: "nth_max {{ {} ; {} ; {} }}".format(t, col, ind),
                   'append': True,
                   'model_args': ['row', 'header', 'n']}

APIs['nth_min'] = {"argument": ['row', 'header', 'num'], 'output': 'obj',
                   "function": lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="min", arg=False),
                   "tostr": lambda t, col, ind: "nth_min {{ {} ; {} ; {} }}".format(t, col, ind),
                   "tosstr": lambda t, col, ind: "nth_min {{ {} ; {} ; {} }}".format(t, col, ind),
                   'append': True,
                   'model_args': ['row', 'header', 'n']}

# With only two argument and the first is not row
APIs['diff'] = {"argument": ['obj', 'obj'], 'output': 'obj',
                'function': lambda t1, t2: obj_compare(t1, t2, type="diff"),
                'tostr': lambda t1, t2: "diff {{ {} ; {} }}".format(t1, t2),
                'tosstr': lambda t1, t2: "diff {{ {} ; {} }}".format(t1, t2),
                'append': True,
                'model_args': ['obj', 'obj']}

# Greater takes two objects of same type (hop on filter_str_eq) and outputs a bool.
# APIs['greater'] = {"argument": ['obj', 'obj'], 'output': 'bool',
#                    'function': lambda t1, t2: obj_compare(t1, t2, type="greater"),
#                    'tostr': lambda t1, t2: "greater {{ {} ; {} }}".format(t1, t2),
#                    'append': False}
#
# APIs['less'] = {"argument": ['obj', 'obj'], 'output': 'bool',
#                 'function': lambda t1, t2: obj_compare(t1, t2, type="less"),
#                 'tostr': lambda t1, t2: "less {{ {} ; {} }}".format(t1, t2),
#                 'append': True}
#
# APIs['eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
#               'function': lambda t1, t2: obj_compare(t1, t2, type="eq"),
#               'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
#               'append': None}
#
# APIs['not_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
#                   'function': lambda t1, t2: obj_compare(t1, t2, type="not_eq"),
#                   'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
#                   "append": None}
#
# APIs['str_eq'] = {"argument": ['str', 'str'], 'output': 'bool',
#                   'function': lambda t1, t2: t1 in t2 or t2 in t1,
#                   'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
#                   "append": None}
#
# APIs['not_str_eq'] = {"argument": ['str', 'str'], 'output': 'bool',
#                       'function': lambda t1, t2: t1 not in t2 and t2 not in t1,
#                       'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
#                       "append": None}
#
# APIs['round_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
#                     'function': lambda t1, t2: obj_compare(t1, t2, round=True, type="eq"),
#                     'tostr': lambda t1, t2: "round_eq {{ {} ; {} }}".format(t1, t2),
#                     'append': None}
#
# APIs['and'] = {"argument": ['bool', 'bool'], 'output': 'bool',
#                'function': lambda t1, t2: t1 and t2,
#                'tostr': lambda t1, t2: "and {{ {} ; {} }}".format(t1, t2),
#                "append": None}

# With only three argument and the first is row
# str
APIs["filter_str_eq"] = {"argument": ['row', 'header_str', 'str'], "output": "row",
                         "function": lambda t, col, value: fuzzy_match_filter(t, col, value),
                         "tostr": lambda t, col, value: "filter_str_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         "tosstr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         'append': False,
                         'model_args': ['row', 'memory_str']}

APIs["filter_str_not_eq"] = {"argument": ['row', 'header_str', 'str'], "output": "row",
                             "function": lambda t, col, value: fuzzy_match_filter(t, col, value, negate=True),
                             "tostr": lambda t, col, value: "filter_str_not_eq {{ {} ; {} ; {} }}".format(t, col,
                                                                                                          value),
                             "tosstr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                             'append': False,
                             'model_args': ['row', 'memory_str']}

# obj: num or str
APIs["filter_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                     "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="eq"),
                     "tostr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                     "tosstr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                     'append': False,
                     'model_args': ['row', 'memory']}

APIs["filter_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                         "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="not_eq"),
                         "tostr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         "tosstr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         'append': False,
                         'model_args': ['row', 'memory']}

APIs["filter_less"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                       "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less"),
                       "tostr": lambda t, col, value: "filter_less {{ {} ; {} ; {} }}".format(t, col, value),
                       "tosstr": lambda t, col, value: "filter_less {{ {} ; {} ; {} }}".format(t, col, value),
                       "append": False,
                       'model_args': ['row', 'memory']}

APIs["filter_greater"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                          "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater"),
                          "tostr": lambda t, col, value: "filter_greater {{ {} ; {} ; {} }}".format(t, col, value),
                          "tosstr": lambda t, col, value: "filter_greater {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": False,
                          'model_args': ['row', 'memory']}

APIs["filter_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                             "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater_eq"),
                             "tostr": lambda t, col, value: "filter_greater_eq {{ {} ; {} ; {} }}".format(t, col,
                                                                                                          value),
                             "tosstr": lambda t, col, value: "filter_greater_eq {{ {} ; {} ; {} }}".format(t, col,
                                                                                                           value),
                             "append": False,
                             'model_args': ['row', 'memory']}

APIs["filter_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                          "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less_eq"),
                          "tostr": lambda t, col, value: "filter_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "tosstr": lambda t, col, value: "filter_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": False,
                          'model_args': ['row', 'memory']}

# APIs["filter_all"] = {"argument": ['row', 'header'], "output": "row",
#                       "function": lambda t, col: t,
#                       "tostr": lambda t, col: "filter_all {{ {} ; {} }}".format(t, col),
#                       'append': False}

# all
# str
# APIs["all_str_eq"] = {"argument": ['row', 'header_str', 'str'], "output": "bool",
#                       "function": lambda t, col, value: len(t) == len(fuzzy_match_filter(t, col, value)),
#                       "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                       "append": None}
#
# APIs["all_str_not_eq"] = {"argument": ['row', 'header_str', 'str'], "output": "bool",
#                           "function": lambda t, col, value: 0 == len(fuzzy_match_filter(t, col, value)),
#                           "tostr": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                           "append": None}
#
# # obj: num or str
# APIs["all_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                   "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="eq")),
#                   "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                   "append": None}
#
# APIs["all_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                       "function": lambda t, col, value: 0 == len(fuzzy_compare_filter(t, col, value, type="eq")),
#                       "tostr": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                       "append": None}
#
# APIs["all_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                     "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="less")),
#                     "tostr": lambda t, col, value: "all_less {{ {} ; {} ; {} }}".format(t, col, value),
#                     "append": None}
#
# APIs["all_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                        "function": lambda t, col, value: len(t) == len(
#                            fuzzy_compare_filter(t, col, value, type="less_eq")),
#                        "tostr": lambda t, col, value: "all_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                        "append": None}
#
# APIs["all_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                        "function": lambda t, col, value: len(t) == len(
#                            fuzzy_compare_filter(t, col, value, type="greater")),
#                        "tostr": lambda t, col, value: "all_greater {{ {} ; {} ; {} }}".format(t, col, value),
#                        "append": None}
#
# APIs["all_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                           "function": lambda t, col, value: len(t) == len(
#                               fuzzy_compare_filter(t, col, value, type="greater_eq")),
#                           "tostr": lambda t, col, value: "all_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                           "append": None}
#
# # most
# # str
# APIs["most_str_eq"] = {"argument": ['row', 'header_str', 'str'], "output": "bool",
#                        "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_match_filter(t, col, value)),
#                        "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                        "append": None}
#
# APIs["most_str_not_eq"] = {"argument": ['row', 'header_str', 'str'], "output": "bool",
#                            "function": lambda t, col, value: len(t) // 3 > len(fuzzy_match_filter(t, col, value)),
#                            "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                            "append": None}
#
# # obj: num or str
# APIs["most_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                    "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="eq")),
#                    "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                    "append": None}
#
# APIs["most_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                        "function": lambda t, col, value: len(t) // 3 > len(
#                            fuzzy_compare_filter(t, col, value, type="eq")),
#                        "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                        "append": None}
#
# APIs["most_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                      "function": lambda t, col, value: len(t) // 3 <= len(
#                          fuzzy_compare_filter(t, col, value, type="less")),
#                      "tostr": lambda t, col, value: "most_less {{ {} ; {} ; {} }}".format(t, col, value),
#                      "append": None}
#
# APIs["most_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                         "function": lambda t, col, value: len(t) // 3 <= len(
#                             fuzzy_compare_filter(t, col, value, type="less_eq")),
#                         "tostr": lambda t, col, value: "most_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                         "append": None}
#
# APIs["most_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                         "function": lambda t, col, value: len(t) // 3 <= len(
#                             fuzzy_compare_filter(t, col, value, type="greater")),
#                         "tostr": lambda t, col, value: "most_greater {{ {} ; {} ; {} }}".format(t, col, value),
#                         "append": None}
#
# APIs["most_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
#                            "function": lambda t, col, value: len(t) // 3 <= len(
#                                fuzzy_compare_filter(t, col, value, type="greater_eq")),
#                            "tostr": lambda t, col, value: "most_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
#                            "append": None}

## Inverse functions here

# Write the greater_str_inv and greater_inv funcs. Takes ['row', 'header_str', 'str', 'header'] and
# ['row', 'header', 'obj', 'header'] respectively as input and returns the header1's values for rows
# whose header2's value is smaller than: hop(filter_eq(header1, $1), header2)
# APIs['greater_inv'] = {"argument": ['row', 'header', 'obj', 'header'], 'output': 'list_obj',
#                        'function': lambda t, col1, val, col2: gl_inv(t, col1, val, col2, "greater"),
#                        'tostr': lambda t, col1, val, col2: "greater_inv {{ {} ; {} ; {} ; {} }}".format(
#                            t, col1, val, col2),
#                        'append': None}

APIs['greater_str_inv'] = {"argument": ['row', 'header_str', 'str', 'header'], 'output': 'list_str',
                           'function': lambda t, col1, val, col2: gl_inv_str(t, col1, val, col2, "greater"),
                           'tostr': lambda t, col1, val, col2: "greater_str_inv {{ {} ; {} ; {} ; {} }}".format(
                               t, col1, val, col2),
                           'tosstr': lambda t, col1, val, col2: "greater_inv {{ {} ; {} ; {} ; {} }}".format(
                               t, col1, val, col2),
                           'append': None,
                           'model_args': ['row', 'memory_str', 'header']}

# APIs['less_inv'] = {"argument": ['row', 'header', 'obj', 'header'], 'output': 'list_obj',
#                     'function': lambda t, col1, val, col2: gl_inv(t, col1, val, col2, "less"),
#                     'tostr': lambda t, col1, val, col2: "less_inv {{ {} ; {} ; {} ; {} }}".format(
#                         t, col1, val, col2),
#                     'append': None}

APIs['less_str_inv'] = {"argument": ['row', 'header_str', 'str', 'header'], 'output': 'list_str',
                        'function': lambda t, col1, val, col2: gl_inv_str(t, col1, val, col2, "less"),
                        'tostr': lambda t, col1, val, col2: "less_str_inv {{ {} ; {} ; {} ; {} }}".format(
                            t, col1, val, col2),
                        'tosstr': lambda t, col1, val, col2: "less_inv {{ {} ; {} ; {} ; {} }}".format(
                            t, col1, val, col2),
                        'append': None,
                        'model_args': ['row', 'memory_str', 'header']}

# Write the most_str_eq_inv func. Takes 'row', 'header' as input and outputs the values which occurs majority
# of the times (>= len(t) // 3). This is not exactly right because the value may also be a substr for majority of
# the columns. But, detecting which set of columns have that substr can be tough. Naive solution requires
# O(nC(t//3) + nC(t//3+1) + ...)
# Precondition: The majority value occurs at least len(t) // 3 times in the header.
APIs["most_str_eq_inv"] = {"argument": ['row', 'header_str'], "output": "list_str",
                           "function": lambda t, col: str_eq_inv(t, col, "most"),
                           "tostr": lambda t, col: "most_str_eq_inv {{ {} ; {} }}".format(t, col),
                           "tosstr": lambda t, col: "most_eq_inv {{ {} ; {} }}".format(t, col),
                           "append": None,
                           'model_args': ['row', 'header_str']}

# Write the all_str_eq_inv func. Takes 'row', 'header' as input and does the following:
# 1. Trim whitespaces 2. Find and return the largest common substr
# Precondition: Such a substr should exist
# TODO: return the largest common substrs
APIs["all_str_eq_inv"] = {"argument": ['row', 'header_str'], "output": "str",
                          "function": lambda t, col: str_eq_inv(t, col, "all"),
                          "tostr": lambda t, col: "all_str_eq_inv {{ {} ; {} }}".format(t, col),
                          "tosstr": lambda t, col: "all_eq_inv {{ {} ; {} }}".format(t, col),
                          "append": None,
                          'model_args': ['row', 'header_str']}

# Write the most_greater_inv func. Takes 'row', 'header' as input, creates a df of datetime and numbers using
# regex pats and returns the largest element x such that len(df[df > x]) >= len(t) // 3. This return value signifies a
# range of values from (-inf, x].
APIs["most_greater_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                                "function": lambda t, col: fuzzy_comp_inv_num(t, col, "mgt"),
                                "tostr": lambda t, col: "most_greater_inv_num {{ {} ; {} }}".format(t, col),
                                "tosstr": lambda t, col: "most_greater_inv_num {{ {} ; {} }}".format(t, col),
                                "append": None,
                                'model_args': ['row', 'header']}

APIs["most_greater_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                                 "function": lambda t, col: fuzzy_comp_inv_date(t, col, "mgt"),
                                 "tostr": lambda t, col: "most_greater_inv_date {{ {} ; {} }}".format(t, col),
                                 "tosstr": lambda t, col: "most_greater_inv_date {{ {} ; {} }}".format(t, col),
                                 "append": None,
                                 'model_args': ['row', 'header']}

APIs["most_greater_eq_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                                   "function": lambda t, col: fuzzy_comp_inv_num(t, col, "mgte"),
                                   "tostr": lambda t, col: "most_greater_eq_inv_num {{ {} ; {} }}".format(t, col),
                                   "tosstr": lambda t, col: "most_greater_eq_inv_num {{ {} ; {} }}".format(t, col),
                                   "append": None,
                                   'model_args': ['row', 'header']}

APIs["most_greater_eq_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                                    "function": lambda t, col: fuzzy_comp_inv_date(t, col, "mgte"),
                                    "tostr": lambda t, col: "most_greater_eq_inv_date {{ {} ; {} }}".format(t, col),
                                    "tosstr": lambda t, col: "most_greater_eq_inv_date {{ {} ; {} }}".format(t, col),
                                    "append": None,
                                    'model_args': ['row', 'header']}

# Write the most_less_inv func. Takes 'row', 'header' as input, creates a df of datetime and numbers using
# regex pats and returns the smallest element x such that len(df[df < x]) >= len(t) // 3. This return value signifies a
# range of values from [x, inf).
APIs["most_less_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                             "function": lambda t, col: fuzzy_comp_inv_num(t, col, "mlt"),
                             "tostr": lambda t, col: "most_less_inv_num {{ {} ; {} }}".format(t, col),
                             "tosstr": lambda t, col: "most_less_inv_num {{ {} ; {} }}".format(t, col),
                             "append": None,
                             'model_args': ['row', 'header']}

APIs["most_less_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                              "function": lambda t, col: fuzzy_comp_inv_date(t, col, "mlt"),
                              "tostr": lambda t, col: "most_less_inv_date {{ {} ; {} }}".format(t, col),
                              "tosstr": lambda t, col: "most_less_inv_date {{ {} ; {} }}".format(t, col),
                              "append": None,
                              'model_args': ['row', 'header']}

APIs["most_less_eq_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                                "function": lambda t, col: fuzzy_comp_inv_num(t, col, "mlte"),
                                "tostr": lambda t, col: "most_less_eq_inv_num {{ {} ; {} }}".format(t, col),
                                "tosstr": lambda t, col: "most_less_eq_inv_num {{ {} ; {} }}".format(t, col),
                                "append": None,
                                'model_args': ['row', 'header']}

APIs["most_less_eq_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                                 "function": lambda t, col: fuzzy_comp_inv_date(t, col, "mlte"),
                                 "tostr": lambda t, col: "most_less_eq_inv_date {{ {} ; {} }}".format(t, col),
                                 "tosstr": lambda t, col: "most_less_eq_inv_date {{ {} ; {} }}".format(t, col),
                                 "append": None,
                                 'model_args': ['row', 'header']}

APIs["all_greater_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                               "function": lambda t, col: fuzzy_comp_inv_num(t, col, "agt"),
                               "tostr": lambda t, col: "all_greater_inv_num {{ {} ; {} }}".format(t, col),
                               "tosstr": lambda t, col: "all_greater_inv_num {{ {} ; {} }}".format(t, col),
                               "append": None,
                               'model_args': ['row', 'header']}

APIs["all_greater_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                                "function": lambda t, col: fuzzy_comp_inv_date(t, col, "agt"),
                                "tostr": lambda t, col: "all_greater_inv_date {{ {} ; {} }}".format(t, col),
                                "tosstr": lambda t, col: "all_greater_inv_date {{ {} ; {} }}".format(t, col),
                                "append": None,
                                'model_args': ['row', 'header']}

APIs["all_greater_eq_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                                  "function": lambda t, col: fuzzy_comp_inv_num(t, col, "agte"),
                                  "tostr": lambda t, col: "all_greater_eq_inv_num {{ {} ; {} }}".format(t, col),
                                  "tosstr": lambda t, col: "all_greater_eq_inv_num {{ {} ; {} }}".format(t, col),
                                  "append": None,
                                  'model_args': ['row', 'header']}

APIs["all_greater_eq_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                                   "function": lambda t, col: fuzzy_comp_inv_date(t, col, "agte"),
                                   "tostr": lambda t, col: "all_greater_eq_inv_date {{ {} ; {} }}".format(t, col),
                                   "tosstr": lambda t, col: "all_greater_eq_inv_date {{ {} ; {} }}".format(t, col),
                                   "append": None,
                                   'model_args': ['row', 'header']}

APIs["all_less_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                            "function": lambda t, col: fuzzy_comp_inv_num(t, col, "alt"),
                            "tostr": lambda t, col: "all_less_inv_num {{ {} ; {} }}".format(t, col),
                            "tosstr": lambda t, col: "all_less_inv_num {{ {} ; {} }}".format(t, col),
                            "append": None,
                            'model_args': ['row', 'header']}

APIs["all_less_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                             "function": lambda t, col: fuzzy_comp_inv_date(t, col, "alt"),
                             "tostr": lambda t, col: "all_less_inv_date {{ {} ; {} }}".format(t, col),
                             "tosstr": lambda t, col: "all_less_inv_date {{ {} ; {} }}".format(t, col),
                             "append": None,
                             'model_args': ['row', 'header']}

APIs["all_less_eq_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                               "function": lambda t, col: fuzzy_comp_inv_num(t, col, "alte"),
                               "tostr": lambda t, col: "all_less_eq_inv_num {{ {} ; {} }}".format(t, col),
                               "tosstr": lambda t, col: "all_less_eq_inv_num {{ {} ; {} }}".format(t, col),
                               "append": None,
                               'model_args': ['row', 'header']}

APIs["all_less_eq_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                                "function": lambda t, col: fuzzy_comp_inv_date(t, col, "alte"),
                                "tostr": lambda t, col: "all_less_eq_inv_date {{ {} ; {} }}".format(t, col),
                                "tosstr": lambda t, col: "all_less_eq_inv_date {{ {} ; {} }}".format(t, col),
                                "append": None,
                                'model_args': ['row', 'header']}

# Write the most_eq_inv func. most_eq_inv takes as input a 'row', 'header' and creates a date df and a num df from
# regex pats. From there, it creates a counter of values and returns all the keys as output that have counts/values
# greater than eq to len(t) // 3
# returns pair of list of obj
APIs["most_eq_inv_num"] = {"argument": ['row', 'header'], "output": "list_num",
                           "function": lambda t, col: fuzzy_comp_inv_num(t, col, "meq"),
                           "tostr": lambda t, col: "most_eq_inv_num {{ {} ; {} }}".format(t, col),
                           "tosstr": lambda t, col: "most_eq_inv_num {{ {} ; {} }}".format(t, col),
                           "append": None,
                           'model_args': ['row', 'header']}

APIs["most_eq_inv_date"] = {"argument": ['row', 'header'], "output": "list_date",
                            "function": lambda t, col: fuzzy_comp_inv_date(t, col, "meq"),
                            "tostr": lambda t, col: "most_eq_inv_date {{ {} ; {} }}".format(t, col),
                            "tosstr": lambda t, col: "most_eq_inv_date {{ {} ; {} }}".format(t, col),
                            "append": None,
                            'model_args': ['row', 'header']}

# Write the all_eq_inv func. Takes 'row', 'header' as input and outputs the eq value(s)
# on datetime df and the num df.
# Precondition: All col values are equal.
APIs["all_eq_inv_num"] = {"argument": ['row', 'header'], "output": "num",
                          "function": lambda t, col: fuzzy_comp_inv_num(t, col, "aeq"),
                          "tostr": lambda t, col: "all_eq_inv_num {{ {} ; {} }}".format(t, col),
                          "tosstr": lambda t, col: "all_eq_inv_num {{ {} ; {} }}".format(t, col),
                          "append": None,
                          'model_args': ['row', 'header']}

APIs["all_eq_inv_date"] = {"argument": ['row', 'header'], "output": "date",
                           "function": lambda t, col: fuzzy_comp_inv_date(t, col, "aeq"),
                           "tostr": lambda t, col: "all_eq_inv_date {{ {} ; {} }}".format(t, col),
                           "tosstr": lambda t, col: "all_eq_inv_date {{ {} ; {} }}".format(t, col),
                           "append": None,
                           'model_args': ['row', 'header']}

month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
             'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9,
             'oct': 10, 'nov': 11, 'dec': 12}
imonth_map = {v: k for k, v in month_map.items()}
imonth_map.update({str(k): v for k, v in imonth_map.items()})

### regex list

# number format:
'''
10
1.12
1,000,000
10:00
1st, 2nd, 3rd, 4th
'''
pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

pat_add = r"((?<==\s)\d+)"

# dates
pat_year = r"\b(\d\d\d\d)\b"
pat_day = r"\b(\d\d?)\b"
pat_month = r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b"


class ExeError(Exception):
    def __init__(self, message=''):
        super(ExeError, self).__init__()
        self.message = message


### for filter functions. we reset index for the result

# filter_str_eq / not_eq
# Type: col - str_col, val - str
def fuzzy_match_filter(t, col, val, negate=False):
    val = str(val)
    t[col] = t[col].astype('str')
    trim_t = t[col].str.replace(" ", "")
    trim_val = val.replace(" ", "")

    if negate:
        res = t[~trim_t.str.contains(trim_val, regex=False)]
    else:
        res = t[trim_t.str.contains(trim_val, regex=False)]
    res = res.reset_index(drop=True)
    return res


# filter nums ...
# Type: col - obj_col, val - obj
def fuzzy_compare_filter(t, col, val, type):
    '''
    fuzzy compare and filter out rows.
    return empty pd if invalid

    type: eq, not_eq, greater, greater_eq, less, less_eq
    '''

    t[col] = t[col].astype('str')
    val = str(val)

    # dates
    if len(re.findall(pat_month, val)) > 0:
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        # print (year_list)
        # print (day_list)
        # print (month_num_list)
        try:
            date_frame = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
        except:
            raise ExeError("Can't create datetime df")
        # print (date_frame)

        # for val
        year_val = re.findall(pat_year, val)
        if len(year_val) == 0:
            year_val = year_list.iloc[0]
        else:
            year_val = int(year_val[0])

        day_val = re.findall(pat_day, val)
        if len(day_val) == 0:
            day_val = day_list.iloc[0]
        else:
            day_val = int(day_val[0])

        month_val = re.findall(pat_month, val)
        if len(month_val) == 0:
            month_val = month_num_list.iloc[0]
        else:
            month_val = month_map[month_val[0]]

        try:
            date_val = pd.datetime(year_val, month_val, day_val)
        except:
            raise ExeError("Can't create datetime val")
        # print (date_val)

        if type == "greater":
            res = t[date_frame > date_val]
        elif type == "greater_eq":
            res = t[date_frame >= date_val]
        elif type == "less":
            res = t[date_frame < date_val]
        elif type == "less_eq":
            res = t[date_frame <= date_val]
        elif type == "eq":
            res = t[date_frame == date_val]
        elif type == "not_eq":
            res = t[~date_frame != date_val]

        res = res.reset_index(drop=True)
        return res

    # numbers, or mixed numbers and strings
    val_pat = re.findall(pat_num, val)
    if len(val_pat) == 0:
        # return pd.DataFrame(columns=list(t.columns))
        # fall back to full string matching
        trim_t = t[col].str.replace(" ", "")
        trim_val = val.replace(" ", "")

        if type == "eq":
            return t[trim_t.str.contains(trim_val, regex=False)].reset_index(drop=True)
        elif type == "not_eq":
            return t[~trim_t.str.contains(trim_val, regex=False)].reset_index(drop=True)
        else:
            return pd.DataFrame(columns=list(t.columns)).reset_index(drop=True)

        # return pd.DataFrame(columns=list(t.columns))

    num = val_pat[0].replace(",", "")
    num = num.replace(":", "")
    num = num.replace(" ", "")
    try:
        num = float(num)
    except:
        num = num.replace(".", "")
        num = float(num)
    # print (num)

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return pd.DataFrame(columns=list(t.columns))
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")
    # print (nums)

    if type == "greater":
        res = t[np.greater(nums, num)]
    elif type == "greater_eq":
        res = t[np.greater_equal(nums, num)]
    elif type == "less":
        res = t[np.less(nums, num)]
    elif type == "less_eq":
        res = t[np.less_equal(nums, num)]
    elif type == "eq":
        res = t[np.isclose(nums, num)]
    elif type == "not_eq":
        res = t[~np.isclose(nums, num)]
    else:
        raise ValueError(f"Unsupported Type: {type}")

    res = res.reset_index(drop=True)
    return res


# For compare inverses
# Type: col - obj_col
def fuzzy_comp_inv_num(t, col, type):
    """
    Fuzzy compare for num

    type: mgt, mlt, mgte, mlte, agt, alt, agte, alte, meq, aeq
    """
    t[col] = t[col].astype('str')
    pats = t[col].str.extract(pat_add, expand=False)

    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)

    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")
    nums = sorted(nums)

    if 'gt' in type:
        if len(t) < 3 and type[0] == 'm':
            raise ExeError(f"Can't apply most on table of length {len(t)}")
        idx = 0 if type[0] == 'a' else len(t) - len(t) // 3
        if 'gte' in type:
            return nums[idx]
        else:
            return nums[idx] - 0.0001
    if 'lt' in type:
        idx = len(t) - 1 if type[0] == 'a' else len(t) // 3 - 1
        if 'lte' in type:
            return nums[idx]
        else:
            return nums[idx] + 0.0001
    if 'eq' in type:
        c_num = Counter(nums)
        if type == 'aeq':
            if not (len(c_num) == 1):
                raise ExeError(f"Unable to apply obj aeq func on col {col}")

            return list(c_num.keys())[0]
        if type == 'meq':
            num_opt, dt_opt = [], []
            for k, v in c_num.items():
                if v >= len(t) // 3:
                    num_opt.append(k)
            if len(num_opt) == 0:
                raise ExeError(f"Unable to apply obj meq func on col {col}")
            return num_opt
    raise ValueError(f"Unsupported type: {type}")


def fuzzy_comp_inv_date(t, col, type):
    """
    Fuzzy compare for date

    type: mgt, mlt, mgte, mlte, agt, alt, agte, alte, meq, aeq
    """
    t[col] = t[col].astype('str')

    # dates
    year_list = t[col].str.extract(pat_year, expand=False)
    day_list = t[col].str.extract(pat_day, expand=False)
    month_list = t[col].str.extract(pat_month, expand=False)
    month_num_list = month_list.map(month_map)
    no_year, no_month, no_day = False, False, False
    if year_list.isnull().any():
        year_list.loc[:] = np.nan
        no_year = True
    if month_num_list.isnull().any():
        month_num_list.loc[:] = np.nan
        no_month = True
    if day_list.isnull().any():
        day_list.loc[:] = np.nan
        no_day = True
    year_ctr = Counter(year_list) if not year_list.isnull().any() else {}
    month_ctr = Counter(month_num_list) if not month_num_list.isnull().any() else {}
    day_ctr = Counter(day_list) if not day_list.isnull().any() else {}

    # pandas at most 2262
    year_list = year_list.fillna("2260").astype("int")
    day_list = day_list.fillna("1").astype("int")
    month_num_list = month_num_list.fillna("1").astype("int")

    try:
        date_frame = pd.to_datetime(
            pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}),
            infer_datetime_format=True)
    except ValueError as err:
        raise ExeError(f"Invalid date values found: {err}")

    date_values = sorted(date_frame)

    if 'gt' in type:
        if len(t) < 3 and type[0] == 'm':
            raise ExeError(f"Can't apply most on table of length {len(t)}")
        idx = 0 if type[0] == 'a' else len(t) - len(t) // 3
        ret_val = date_values[idx] if 'gte' in type else date_values[idx] - pd.Timedelta(days=1)
        date_ret_val = '' if no_day else 'd'
        date_ret_val += '' if no_month else 'm'
        date_ret_val += '' if no_year else 'y'
        date_ret_val += ': '
        date_ret_val += '' if no_day else f'{ret_val.day} '
        date_ret_val += '' if no_month else f'{imonth_map[ret_val.month]} '
        date_ret_val += '' if no_year else f'{ret_val.year} '
        date_ret_val = date_ret_val[:-1]
        return date_ret_val
    if 'lt' in type:
        idx = len(t) - 1 if type[0] == 'a' else len(t) // 3 - 1
        ret_val = date_values[idx] if 'lte' in type else date_values[idx] + pd.Timedelta(days=1)
        date_ret_val = '' if no_day else 'd'
        date_ret_val += '' if no_month else 'm'
        date_ret_val += '' if no_year else 'y'
        date_ret_val += ': '
        date_ret_val += '' if no_day else f'{ret_val.day} '
        date_ret_val += '' if no_month else f'{imonth_map[ret_val.month]} '
        date_ret_val += '' if no_year else f'{ret_val.year} '
        date_ret_val = date_ret_val[:-1]
        return date_ret_val
    if 'eq' in type:
        if type == 'aeq':
            if not (len(year_ctr) == 1 or len(month_ctr) == 1 or len(day_ctr) == 1):
                raise ExeError(f"Unable to apply obj aeq func on col {col}")

            def get_val(ctr):
                return list(ctr.keys())[0]

            date_ret_val = '' if not len(day_ctr) == 1 else 'd'
            date_ret_val += '' if not len(month_ctr) == 1 else 'm'
            date_ret_val += '' if not len(year_ctr) == 1 else 'y'
            date_ret_val += ': '
            date_ret_val += '' if not len(day_ctr) == 1 else f'{get_val(day_ctr)} '
            date_ret_val += '' if not len(month_ctr) == 1 else f'{imonth_map[get_val(month_ctr)]} '
            date_ret_val += '' if not len(year_ctr) == 1 else f'{get_val(year_ctr)} '
            date_ret_val = date_ret_val[:-1]
            return date_ret_val
        if type == 'meq':
            dt_opt = []
            for fmt in powerset('dmy'):
                if len(fmt) == 0:
                    continue
                if 'd' in fmt and no_day:
                    continue
                if 'm' in fmt and no_month:
                    continue
                if 'y' in fmt and no_year:
                    continue
                tot_rows = t.shape[0]
                dl = day_list if 'd' in fmt else [1] * tot_rows
                ml = month_num_list if 'm' in fmt else [1] * tot_rows
                yl = year_list if 'y' in fmt else [2260] * tot_rows
                _df = pd.to_datetime(
                    pd.DataFrame({'year': yl, 'month': ml, 'day': dl}),
                    infer_datetime_format=True)
                _dv = sorted(_df)
                c_dt = Counter(_dv)
                for k, v in c_dt.items():
                    if v >= len(t) // 3:
                        val = '' if not 'd' in fmt else 'd'
                        val += '' if not 'm' in fmt else 'm'
                        val += '' if not 'y' in fmt else 'y'
                        val += ': '
                        val += '' if not 'd' in fmt else f'{k.day} '
                        val += '' if not 'm' in fmt else f'{imonth_map[k.month]} '
                        val += '' if not 'y' in fmt else f'{k.year} '
                        val = val[:-1]
                        dt_opt.append(val)
            if len(dt_opt) == 0:
                raise ExeError(f"Unable to apply obj meq func on col {col}")
            return dt_opt
    raise ValueError(f"Unsupported type: {type}")


### for comparison
# Type: obj
def safe_obj_compare(obj1, obj2, type, only):
    try:
        return obj_compare(obj1, obj2, type=type, only=only)
    except ExeError:
        return False


def obj_compare(num1, num2, round=False, type="eq", only=''):
    tolerance = 0.15 if round else 1e-9
    # both numeric
    if isinstance(num1, (list, tuple)) or isinstance(num2, (list, tuple)):
        raise ValueError(f"Bad input to obj_compare num1: {num1}, num2: {num2}")
    try:
        if only == 'date':
            raise ValueError()
        num_1 = float(num1)
        num_2 = float(num2)

        # if negate:
        #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        # return math.isclose(num_1, num_2, rel_tol=tolerance)

        if type == "eq":
            return math.isclose(num_1, num_2, rel_tol=tolerance)
        elif type == "not_eq":
            return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        elif type == "greater":
            return num_1 > num_2
        elif type == "less":
            return num_1 < num_2
        elif type == "diff":
            return num_1 - num_2
        elif type == "greater_eq":
            return num_1 >= num_2
        elif type == "less_eq":
            return num_1 <= num_2
    except ValueError:
        # strings
        # mixed numbers and strings
        # num1 is actual value
        num1 = str(num1)
        # num2 is the returned value
        num2 = str(num2)

        # dates
        # num1
        if len(re.findall(pat_month, num1)) > 0:
            try:
                if only == 'num':
                    raise ValueError()
                comp_y, comp_m, comp_d = False, False, False
                year_val1 = re.findall(pat_year, num1)
                if len(year_val1) == 0:
                    year_val1 = int("2260")
                else:
                    year_val1 = int(year_val1[0])
                    comp_y = True

                day_val1 = re.findall(pat_day, num1)
                if len(day_val1) == 0:
                    day_val1 = int("1")
                else:
                    day_val1 = int(day_val1[0])
                    comp_d = True

                month_val1 = re.findall(pat_month, num1)
                if len(month_val1) == 0:
                    month_val1 = int("1")
                else:
                    month_val1 = month_map[month_val1[0]]
                    comp_m = True

                if not (comp_d or comp_m or comp_y):
                    raise ExeError(f"Actual val has no date parts: {num1}")

                try:
                    date_val1 = pd.datetime(year_val1, month_val1, day_val1)
                except:
                    raise ExeError(f"Unable to convert val to datetime: {num1}")

                # num2
                year_val2 = re.findall(pat_year, num2)
                if len(year_val2) == 0 and comp_y:
                    raise ExeError(f"Year val not present in returned: {num2}")
                elif len(year_val2) == 0 or not comp_y:
                    year_val2 = int("2260")
                else:
                    year_val2 = int(year_val2[0])

                day_val2 = re.findall(pat_day, num2)
                if len(day_val2) == 0 and comp_d:
                    raise ExeError(f"Day val not present in returned: {num2}")
                elif len(day_val2) == 0 or not comp_d:
                    day_val2 = int("1")
                else:
                    day_val2 = int(day_val2[0])

                month_val2 = re.findall(pat_month, num2)
                if len(month_val2) == 0 and comp_m:
                    raise ExeError(f"Month val not present in returned: {num2}")
                elif len(month_val2) == 0 or not comp_m:
                    month_val2 = int("1")
                else:
                    month_val2 = month_map[month_val2[0]]

                try:
                    date_val2 = pd.datetime(year_val2, month_val2, day_val2)
                except:
                    raise ExeError(f"Unable to convert val to datetime: {num2}")

                # if negate:
                #   return date_val1 != date_val2
                # else:
                #   return date_val1 == date_val2

                if type == "eq":
                    return date_val1 == date_val2
                elif type == "not_eq":
                    return date_val1 != date_val2
                elif type == "greater":
                    return date_val1 > date_val2
                elif type == "less":
                    return date_val1 < date_val2
                # for diff return string
                elif type == "diff":
                    return str((date_val1 - date_val2).days) + " days"
                elif type == "greater_eq":
                    return date_val1 >= date_val2
                elif type == "less_eq":
                    return date_val1 <= date_val2
            except:
                pass

        # mixed string and numerical
        if only == 'date':
            raise ExeError()
        val_pat1 = re.findall(pat_num, num1)
        val_pat2 = re.findall(pat_num, num2)
        if len(val_pat1) == 0 or len(val_pat2) == 0:
            # fall back to full string matching
            num1 = num1.replace(" ", "")
            num2 = num2.replace(" ", "")
            if type == "not_eq":
                return (num1 not in num2) and (num2 not in num1)
            elif type == "eq":
                return num1 in num2 or num2 in num1
            else:
                raise ExeError(f"Unsupported type: {type}")

        num_1 = val_pat1[0].replace(",", "")
        num_1 = num_1.replace(":", "")
        num_1 = num_1.replace(" ", "")
        try:
            num_1 = float(num_1)
        except:
            num_1 = num_1.replace(".", "")
            num_1 = float(num_1)

        num_2 = val_pat2[0].replace(",", "")
        num_2 = num_2.replace(":", "")
        num_2 = num_2.replace(" ", "")
        try:
            num_2 = float(num_2)
        except:
            num_2 = num_2.replace(".", "")
            num_2 = float(num_2)

        # if negate:
        #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        # return math.isclose(num_1, num_2, rel_tol=tolerance)

        if type == "eq":
            return math.isclose(num_1, num_2, rel_tol=tolerance)
        elif type == "not_eq":
            return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        elif type == "greater":
            return num_1 > num_2
        elif type == "less":
            return num_1 < num_2
        elif type == "diff":
            return num_1 - num_2
        elif type == "greater_eq":
            return num_1 >= num_2
        elif type == "less_eq":
            return num_1 <= num_2
        else:
            raise ValueError(f"Unsupported type: {type}")


### for aggregation: sum avg
# Type: num
def agg(t, col, type):
    '''
    sum or avg for aggregation
    '''

    # unused
    if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
        if type == "sum":
            res = t[col].sum()
        elif type == "avg":
            res = t[col].mean()
        else:
            raise ValueError(f"Unsupported Type: {type}")
        return res

    else:

        pats = t[col].str.extract(pat_add, expand=False)
        if pats.isnull().all():
            pats = t[col].str.extract(pat_num, expand=False)
        if pats.isnull().all():
            return 0.0
        pats.fillna("0.0")
        nums = pats.str.replace(",", "")
        nums = nums.str.replace(":", "")
        nums = nums.str.replace(" ", "")
        try:
            nums = nums.astype("float")
        except:
            nums = nums.str.replace(".", "")
            nums = nums.astype("float")

        # print (nums)
        if type == "sum":
            return nums.sum()
        elif type == "avg":
            return nums.mean()
        else:
            raise ValueError(f"Unsupported Type: {type}")


### for hop

def hop_op(t, col):
    if len(t) == 0:
        raise ValueError("Hop received 0 len df")

    return t[col].values[0]


### for superlative, ordinal
# Type: col - obj
def nth_maxmin(t, col, order=1, max_or_min="max", arg=False):
    '''
    for max, min, argmax, argmin,
    nth_max, nth_min, nth_argmax, nth_argmin

    return string or rows
    '''
    t[col] = t[col].astype('str')

    order = int(order)
    ### return the original content for max,min
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        # print (year_list)
        # print (day_list)
        # print (month_num_list)

        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            # print (date_series)

            if max_or_min == "max":
                tar_row = date_series.nlargest(order).iloc[[-1]]
            elif max_or_min == "min":
                tar_row = date_series.nsmallest(order).iloc[[-1]]
            ind = list(tar_row.index.values)
            if arg:
                res = t.iloc[ind]
            else:
                res = t.iloc[ind][col].values[0]

            return res

        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        raise ExeError(f"df col {col} not a obj type")
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    try:
        if max_or_min == "max":
            tar_row = nums.nlargest(order).iloc[[-1]]
        elif max_or_min == "min":
            tar_row = nums.nsmallest(order).iloc[[-1]]
        ind = list(tar_row.index.values)
        # print (ind)
        # print (t.iloc[ind][col].values)
        if arg:
            res = t.iloc[ind]
        else:
            res = t.iloc[ind][col].values[0]
    except Exception as e:
        raise e

    return res


def str_eq_inv(t, col, type):
    """
    type: all, most
    """
    if type == 'all':
        def get_lcs(a, b):
            a, b = str(a), str(b)
            from difflib import SequenceMatcher
            match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
            return a[match.a:match.a + match.size].strip()

        # if len(ctr) != 1:
        #     raise ExeError(f"Can't apply str aeq on col {col}")
        # return list(ctr.keys())[0]
        lcs = t[col].iloc[0]
        for s in t[col].tolist():
            lcs = get_lcs(lcs, s)
            if len(lcs) == 0:
                break
        if len(lcs) == 0:
            raise ExeError(f"Can't apply str aeq on col {col}")
        return lcs
    if type == 'most':
        ctr = Counter(t[col])
        vals = []
        for k, v in ctr.items():
            if v >= len(t) // 3:
                vals.append(k)
        if len(vals) == 0:
            raise ExeError(f"Can't apply str meq on col {col}")
        return vals
    raise ValueError(f"Unsupported Type: {type}")


def gl_inv(t, col1, val, col2, type):
    """
    type: greater, less
    """
    df = fuzzy_compare_filter(t, col1, val, type="eq")
    if len(df) != 1:
        raise ExeError(f"Can't apply obj-{type}-inv for the filter col {col1} and val {val}")
    tmp1 = hop_op(df, col2)
    ret_vals = []
    for tmp2, hop_val in zip(t[col2], t[col1]):
        if obj_compare(tmp1, tmp2, type=type):
            ret_vals.append(hop_val)
    if len(ret_vals) == 0:
        raise ExeError(f"No rows found {type} than {tmp1} in col {col2}")
    return ret_vals


def gl_inv_str(t, col1, val, col2, type):
    """
    type: greater, less
    """
    t[col1] = t[col1].astype('str')
    df = fuzzy_match_filter(t, col1, val)
    if len(df) != 1:
        raise ExeError(f"Can't apply obj-{type}-inv for the filter col {col1} and val {val}")
    tmp1 = hop_op(df, col2)
    ret_vals = []
    for tmp2, hop_val in zip(t[col2], t[col1]):
        try:
            if obj_compare(tmp1, tmp2, type=type):
                ret_vals.append(hop_val)
        except:
            pass
    if len(ret_vals) == 0:
        raise ExeError(f"No rows found {type} than {tmp1} in col {col2}")
    return ret_vals


def check_if_accept(func, returned, actual):
    v = APIs[func]
    k = func
    if k in ['avg', 'sum']:
        return obj_compare(actual, returned, True)
    elif v['argument'] == ['row', 'header']:
        if 'greater' in k:
            comp_type = "less_eq"
        elif 'less' in k:
            comp_type = "greater_eq"
        elif k in ['all_eq_inv_num', 'all_eq_inv_date']:
            comp_type = "eq"
        else:
            return False
        if v['output'] == 'date':
            if len(re.findall(pat_month, actual)) > 0:
                if returned is None:
                    return False
                return safe_obj_compare(actual, returned, type=comp_type, only='date')
        elif v['output'] == 'num':
            if len(re.findall(pat_num, actual)) > 0:
                if returned is None:
                    return False
                return safe_obj_compare(actual, returned, type=comp_type, only='num')
    elif v['argument'] == ['row', 'header'] and v['output'] == 'list_num':
        if len(re.findall(pat_num, actual)) > 0:
            if returned is None:
                return False
            return any(safe_obj_compare(actual, ret_val, 'eq', 'num') for ret_val in returned)
    elif v['argument'] == ['row', 'header'] and v['output'] == 'list_date':
        if len(re.findall(pat_month, actual)) > 0:
            if returned is None:
                return False
            return any(safe_obj_compare(actual, ret_val, 'eq', 'date') for ret_val in returned)
    elif v['output'] in ['num', 'obj', 'str']:
        return obj_compare(actual, returned)
    elif v['output'] == 'list_str':
        return any(obj_compare(actual, ret_val) for ret_val in returned)

    return False


sharper_triggers = {}
sharper_triggers['avg'] = ['average']

sharper_triggers['diff'] = ['after', 'before', 'difference', 'gap', 'than', 'separate', 'except', 'but', 'separation']
sharper_triggers['add'] = ['sum', 'summation', 'combine', 'combined', 'total', 'add', 'all', 'there are']
sharper_triggers['sum'] = sharper_triggers['add']

sharper_triggers['not_eq'] = ['not', 'no', 'never', "didn't", "won't", "wasn't", "isn't",
                              "haven't", "weren't", "won't", 'neither', 'none', 'unable',
                              'fail', 'different', 'outside', 'unable', 'fail']

sharper_triggers['filter_str_not_eq'] = sharper_triggers['not_eq']
sharper_triggers['filter_not_eq'] = sharper_triggers['not_eq']

sharper_triggers["filter_greater"] = ['higher', 'greater', 'more', 'than', 'above', 'after', 'through', 'to', 'between']
sharper_triggers["filter_less"] = ['lower', 'earlier', 'less', 'than', 'below', 'under', 'before',
                                   'through', 'to', 'between']
sharper_triggers['less'] = ['lower', 'earlier', 'less', 'than', 'before', 'below', 'under']
sharper_triggers['greater'] = ['higher', 'greater', 'more', 'than', 'above', 'after', 'exceed', 'over']

sharper_triggers['all_eq'] = ['all', 'every', 'each', 'only', 'always']
sharper_triggers['all_less'] = [['all', 'every', 'each'], sharper_triggers['less']]
sharper_triggers['all_greater'] = [['all', 'every', 'each'], sharper_triggers['greater']]
sharper_triggers['all_str_eq'] = ['all', 'every', 'each', 'always']

sharper_triggers["all_str_not_eq"] = [['all', 'every', 'each', 'always'],
                                      ['not', 'no', 'never', "didn't", "won't", "wasn't"]]
sharper_triggers["all_not_eq"] = sharper_triggers["all_str_not_eq"]

sharper_triggers['filter_less_eq'] = ['at most']
sharper_triggers['filter_greater_eq'] = ['at least']

sharper_triggers['all_less_eq'] = [sharper_triggers['filter_less_eq'], ['all', 'while', 'every', 'each']]
sharper_triggers['all_greater_eq'] = [sharper_triggers['filter_greater_eq'], ['all', 'while', 'every', 'each']]

sharper_triggers['max'] = ['highest', 'most', 'greatest', 'maximum', 'max']
sharper_triggers['min'] = ['lowest', 'earliest', 'least', 'smallest', 'minimum', 'min']

sharper_triggers['argmax'] = ['highest', 'most', 'greatest', 'maximum', 'max', 'top', 'first']
sharper_triggers['argmin'] = ['lowest', 'earliest', 'least', 'smallest', 'minimum', 'min', 'bottom', 'last']

# Inverse non triggers
sharper_triggers['greater_str_inv'] = sharper_triggers['greater']
sharper_triggers['less_str_inv'] = sharper_triggers['less']

sharper_triggers['most_eq_inv_date'] = sharper_triggers['most_eq_inv_num'] = ['most', 'majority']
sharper_triggers['most_str_eq_inv'] = sharper_triggers['most_eq_inv_date']
sharper_triggers['most_greater_inv_date'] = sharper_triggers['most_greater_inv_num'] = sharper_triggers[
    'most_eq_inv_date']
sharper_triggers['most_greater_eq_inv_date'] = sharper_triggers['most_greater_eq_inv_num'] = [
    sharper_triggers['filter_greater_eq'], sharper_triggers['most_eq_inv_date']]
sharper_triggers['most_less_inv_date'] = sharper_triggers['most_less_inv_num'] = [
    sharper_triggers["filter_less"], sharper_triggers['most_eq_inv_date']]
sharper_triggers['most_less_eq_inv_date'] = sharper_triggers['most_less_eq_inv_num'] = sharper_triggers[
    'most_eq_inv_date']

sharper_triggers['all_str_eq_inv'] = sharper_triggers['all_str_eq']
sharper_triggers['all_greater_inv_date'] = sharper_triggers['all_greater_inv_num'] = sharper_triggers['all_greater']
sharper_triggers['all_greater_eq_inv_date'] = sharper_triggers['all_greater_eq_inv_num'] = sharper_triggers[
    'all_greater_eq']
sharper_triggers['all_less_inv_date'] = sharper_triggers['all_less_inv_num'] = sharper_triggers['all_less']
sharper_triggers['all_less_eq_inv_date'] = sharper_triggers['all_less_eq_inv_num'] = sharper_triggers['all_less_eq']
sharper_triggers['all_eq_inv_date'] = sharper_triggers['all_eq_inv_num'] = sharper_triggers['all_eq']
