#=======================================================================================================
# Tegridy MIDI Helper functions
#=======================================================================================================

import os
import math
from collections import Counter
from itertools import product, groupby

#=======================================================================================================

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]

#===============================================================================

def generate_colors(n):
    return [hsv_to_rgb(i/n, 1, 1) for i in range(n)]

#===============================================================================

def add_arrays(a, b):
    return [sum(pair) for pair in zip(a, b)]

#===============================================================================

def group_sublists_by_length(lst):
    unique_lengths = sorted(list(set(map(len, lst))), reverse=True)
    return [[x for x in lst if len(x) == i] for i in unique_lengths]

#===============================================================================

def stack_list(lst, base=12):
    return sum(j * base**i for i, j in enumerate(lst[::-1]))

#===============================================================================

def destack_list(num, base=12):
    lst = []
    while num:
        lst.append(num % base)
        num //= base
    return lst[::-1]

#===============================================================================

def split_list(list_to_split, split_value=0):
    
    size = len(list_to_split)
    idx_list = [idx + 1 for idx, val in
                enumerate(list_to_split) if val == split_value]


    res = [list_to_split[i: j] for i, j in
            zip([0] + idx_list, idx_list + 
            ([size] if idx_list[-1] != size else []))]
  
    return res

#===============================================================================

def find_exact_match_variable_length(list_of_lists, target_list, uncertain_indices):

    possible_values = {idx: set() for idx in uncertain_indices}
    for sublist in list_of_lists:
        for idx in uncertain_indices:
            if idx < len(sublist):
                possible_values[idx].add(sublist[idx])
    
    uncertain_combinations = product(*(possible_values[idx] for idx in uncertain_indices))
    
    for combination in uncertain_combinations:

        test_list = target_list[:]
        for idx, value in zip(uncertain_indices, combination):
            test_list[idx] = value
        
        for sublist in list_of_lists:
            if len(sublist) >= len(test_list) and sublist[:len(test_list)] == test_list:
                return sublist  # Return the matching sublist
    
    return None  # No exact match found

#===============================================================================

def ceil_with_precision(value, decimal_places):
    factor = 10 ** decimal_places
    return math.ceil(value * factor) / factor

#===============================================================================

def grouped_set(seq):
  return [k for k, v in groupby(seq)]

#===============================================================================

def ordered_set(seq):
  dic = {}
  return [k for k, v in dic.fromkeys(seq).items()]

#===============================================================================

def find_paths(list_of_lists, path=[]):
    if not list_of_lists:
        return [path]
    return [p for sublist in list_of_lists[0] for p in find_paths(list_of_lists[1:], path+[sublist])]

#===============================================================================

def flatten(list_of_lists):
  return [x for y in list_of_lists for x in y]

#===============================================================================

def sort_list_by_other(list1, list2):
    return sorted(list1, key=lambda x: list2.index(x) if x in list2 else len(list2))

#===============================================================================

def find_closest_value(lst, val):

  closest_value = min(lst, key=lambda x: abs(val - x))
  closest_value_indexes = [i for i in range(len(lst)) if lst[i] == closest_value]
  
  return [closest_value, abs(val - closest_value), closest_value_indexes]

#===============================================================================

def count_patterns(lst, sublist):
    count = 0
    idx = 0
    for i in range(len(lst) - len(sublist) + 1):
        if lst[idx:idx + len(sublist)] == sublist:
            count += 1
            idx += len(sublist)
        else:
          idx += 1
    return count

#===============================================================================

def find_lrno_patterns(seq):

  all_seqs = Counter()

  max_pat_len = math.ceil(len(seq) / 2)

  num_iter = 0

  for i in range(len(seq)):
    for j in range(i+1, len(seq)+1):
      if j-i <= max_pat_len:
        all_seqs[tuple(seq[i:j])] += 1
        num_iter += 1

  max_count = 0
  max_len = 0

  for val, count in all_seqs.items():

    if max_len < len(val):
      max_count = max(2, count)

    if count > 1:
      max_len = max(max_len, len(val))
      pval = val

  max_pats = []

  for val, count in all_seqs.items():
    if count == max_count and len(val) == max_len:
      max_pats.append(val)

  found_patterns = []

  for pat in max_pats:
    count = count_patterns(seq, list(pat))
    if count > 1:
      found_patterns.append([count, len(pat), pat])

  return found_patterns

#===============================================================================

def split_list(lst, val):
    return [lst[i:j] for i, j in zip([0] + [k + 1 for k, x in enumerate(lst) if x == val], [k for k, x in enumerate(lst) if x == val] + [len(lst)]) if j > i]

#===============================================================================

def adjust_numbers_to_sum(numbers, target_sum):

  current_sum = sum(numbers)
  difference = target_sum - current_sum

  non_zero_elements = [(i, num) for i, num in enumerate(numbers) if num != 0]

  total_non_zero = sum(num for _, num in non_zero_elements)

  increments = []
  for i, num in non_zero_elements:
      proportion = num / total_non_zero
      increment = proportion * difference
      increments.append(increment)

  for idx, (i, num) in enumerate(non_zero_elements):
      numbers[i] += int(round(increments[idx]))

  current_sum = sum(numbers)
  difference = target_sum - current_sum
  non_zero_indices = [i for i, num in enumerate(numbers) if num != 0]

  for i in range(abs(difference)):
      numbers[non_zero_indices[i % len(non_zero_indices)]] += 1 if difference > 0 else -1

  return numbers

#===============================================================================

def horizontal_ordered_list_search(list_of_lists, 
                                    query_list, 
                                    start_idx=0,
                                    end_idx=-1
                                    ):

  lol = list_of_lists

  results = []

  if start_idx > 0:
    lol = list_of_lists[start_idx:]

  if start_idx == -1:
    idx = -1
    for i, l in enumerate(list_of_lists):
      try:
        idx = l.index(query_list[0])
        lol = list_of_lists[i:]
        break
      except:
        continue

    if idx == -1:
      results.append(-1)
      return results
    else:
      results.append(i)

  if end_idx != -1:
    lol = list_of_lists[start_idx:start_idx+max(end_idx, len(query_list))]

  for i, q in enumerate(query_list):
    try:
      idx = lol[i].index(q)
      results.append(idx)
    except:
      results.append(-1)
      return results

  return results

#===============================================================================

def ordered_lists_match_ratio(src_list, trg_list):

  zlist = list(zip(src_list, trg_list))

  return sum([a == b for a, b in zlist]) / len(list(zlist))

#===============================================================================

def lists_intersections(src_list, trg_list):
  return list(set(src_list) & set(trg_list))

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of helpers module
#=======================================================================================================
