#=======================================================================================================
# Tegridy MIDI Helper functions
#=======================================================================================================

import os
import math
import copy
from collections import Counter
from itertools import product, groupby

import numpy as np

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

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

def lists_sym_differences(src_list, trg_list):
  return list(set(src_list) ^ set(trg_list))

#===============================================================================

def lists_differences(long_list, short_list):
  return list(set(long_list) - set(short_list))

#===============================================================================

def adjust_list_of_values_to_target_average(list_of_values, 
                                            trg_avg, 
                                            min_value, 
                                            max_value
                                            ):

    filtered_values = [value for value in list_of_values if min_value <= value <= max_value]

    if not filtered_values:
        return list_of_values

    current_avg = sum(filtered_values) / len(filtered_values)
    scale_factor = trg_avg / current_avg

    adjusted_values = [value * scale_factor for value in filtered_values]

    total_difference = trg_avg * len(filtered_values) - sum(adjusted_values)
    adjustment_per_value = total_difference / len(filtered_values)

    final_values = [value + adjustment_per_value for value in adjusted_values]

    while abs(sum(final_values) / len(final_values) - trg_avg) > 1e-6:
        total_difference = trg_avg * len(final_values) - sum(final_values)
        adjustment_per_value = total_difference / len(final_values)
        final_values = [value + adjustment_per_value for value in final_values]

    final_values = [round(value) for value in final_values]

    adjusted_values = copy.deepcopy(list_of_values)

    j = 0

    for i in range(len(adjusted_values)):
        if min_value <= adjusted_values[i] <= max_value:
            adjusted_values[i] = final_values[j]
            j += 1

    return adjusted_values

#===============================================================================

def minkowski_distance(x, y, p=3, pad_value=float('inf')):

    if len(x) != len(y):
      return -1
    
    distance = 0
    
    for i in range(len(x)):

        if x[i] == pad_value or y[i] == pad_value:
          continue

        distance += abs(x[i] - y[i]) ** p

    return distance ** (1 / p)

#===============================================================================

def dot_product(x, y, pad_value=None):
    return sum(xi * yi for xi, yi in zip(x, y) if xi != pad_value and yi != pad_value)

#===============================================================================

def norm(vector, pad_value=None):
    return sum(xi ** 2 for xi in vector if xi != pad_value) ** 0.5

#===============================================================================

def cosine_similarity(x, y, pad_value=None):
    if len(x) != len(y):
        return -1
    
    dot_prod = dot_product(x, y, pad_value)
    norm_x = norm(x, pad_value)
    norm_y = norm(y, pad_value)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    return dot_prod / (norm_x * norm_y)

#===============================================================================

def hamming_distance(arr1, arr2, pad_value):
    return sum(el1 != el2 for el1, el2 in zip(arr1, arr2) if el1 != pad_value and el2 != pad_value)

#===============================================================================

def jaccard_similarity(arr1, arr2, pad_value):
    intersection = sum(el1 and el2 for el1, el2 in zip(arr1, arr2) if el1 != pad_value and el2 != pad_value)
    union = sum((el1 or el2) for el1, el2 in zip(arr1, arr2) if el1 != pad_value or el2 != pad_value)
    return intersection / union if union != 0 else 0

#===============================================================================

def pearson_correlation(arr1, arr2, pad_value):
    filtered_pairs = [(el1, el2) for el1, el2 in zip(arr1, arr2) if el1 != pad_value and el2 != pad_value]
    if not filtered_pairs:
        return 0
    n = len(filtered_pairs)
    sum1 = sum(el1 for el1, el2 in filtered_pairs)
    sum2 = sum(el2 for el1, el2 in filtered_pairs)
    sum1_sq = sum(el1 ** 2 for el1, el2 in filtered_pairs)
    sum2_sq = sum(el2 ** 2 for el1, el2 in filtered_pairs)
    p_sum = sum(el1 * el2 for el1, el2 in filtered_pairs)
    num = p_sum - (sum1 * sum2 / n)
    den = ((sum1_sq - sum1 ** 2 / n) * (sum2_sq - sum2 ** 2 / n)) ** 0.5
    if den == 0:
        return 0
    return num / den

#===============================================================================

def find_value_power(value, number):
    return math.floor(math.log(value, number))

#===============================================================================

def proportions_counter(list_of_values):

  counts = Counter(list_of_values).most_common()
  clen = sum([c[1] for c in counts])

  return [[c[0], c[1], c[1] / clen] for c in counts]

#===============================================================================

def find_pattern_start_indexes(values, pattern):

  start_indexes = []

  count = 0

  for i in range(len(values)- len(pattern)):
    chunk = values[i:i+len(pattern)]

    if chunk == pattern:
      start_indexes.append(i)

  return start_indexes

#===============================================================================

def all_consequtive(list_of_values):
  return all(b > a for a, b in zip(list_of_values[:-1], list_of_values[1:]))

#===============================================================================

def calculate_similarities(lists_of_values, metric='cosine'):
  return metrics.pairwise_distances(lists_of_values, metric=metric).tolist()

#===============================================================================

def get_tokens_embeddings(x_transformer_model):
  return x_transformer_model.net.token_emb.emb.weight.detach().cpu().tolist()

#===============================================================================

def minkowski_distance_matrix(X, p=3):

  X = np.array(X)

  n = X.shape[0]
  dist_matrix = np.zeros((n, n))

  for i in range(n):
      for j in range(n):
          dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j])**p)**(1/p)

  return dist_matrix.tolist()

#===============================================================================

def robust_normalize(values):

  values = np.array(values)
  q1 = np.percentile(values, 25)
  q3 = np.percentile(values, 75)
  iqr = q3 - q1

  filtered_values = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]

  min_val = np.min(filtered_values)
  max_val = np.max(filtered_values)
  normalized_values = (values - min_val) / (max_val - min_val)

  normalized_values = np.clip(normalized_values, 0, 1)

  return normalized_values.tolist()

#===============================================================================

def min_max_normalize(values):

  scaler = MinMaxScaler()

  return scaler.fit_transform(values).tolist()

#===============================================================================

def normalize_to_range(values, n):
    
  min_val = min(values)
  max_val = max(values)
  
  range_val = max_val - min_val
  
  normalized_values = [((value - min_val) / range_val * 2 * n) - n for value in values]
  
  return normalized_values

#===============================================================================

def filter_and_replace_values(list_of_values, 
                              threshold, 
                              replace_value, 
                              replace_above_threshold=False
                              ):

  array = np.array(list_of_values)

  modified_array = np.copy(array)
  
  if replace_above_threshold:
    modified_array[modified_array > threshold] = replace_value
  
  else:
    modified_array[modified_array < threshold] = replace_value
  
  return modified_array.tolist()

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of helpers module
#=======================================================================================================
