#=======================================================================================================
# Tegridy MIDI Miscellaneous module
#=======================================================================================================

import copy
import statistics
from collections import Counter

from tegridymidi.processors import recalculate_score_timings, delta_score_notes, delta_score_to_abs_score
from tegridymidi.chords import chordify_score, pitches_to_tones_chord, chord_to_pchord, advanced_check_and_fix_chords_in_chordified_score
from tegridymidi.chords import chordified_score_pitches
from tegridymidi.constants import Number2patch, MIDI_Instruments_Families, ALL_CHORDS, ALL_CHORDS_FULL
from tegridymidi.constants import WHITE_NOTES, BLACK_NOTES, CHORDS_TYPES
from tegridymidi.helpers import flatten, adjust_numbers_to_sum, hamming_distance, jaccard_similarity, pearson_correlation
from tegridymidi.helpers import find_pattern_start_indexes, find_lrno_patterns
from tegridymidi.matrixes import escore_notes_to_escore_matrix, escore_matrix_to_original_escore_notes

#===============================================================================

def adjust_score_velocities(score, max_velocity):

    min_velocity = min([c[5] for c in score])
    max_velocity_all_channels = max([c[5] for c in score])
    min_velocity_ratio = min_velocity / max_velocity_all_channels

    max_channel_velocity = max([c[5] for c in score])
    
    if max_channel_velocity < min_velocity:
        factor = max_velocity / min_velocity
    else:
        factor = max_velocity / max_channel_velocity
    for i in range(len(score)):
        score[i][5] = int(score[i][5] * factor)

#===============================================================================

def analyze_score_pitches(score, channels_to_analyze=[0]):

  analysis = {}

  score_notes = [s for s in score if s[3] in channels_to_analyze]

  cscore = chordify_score(score_notes)

  chords_tones = []

  all_tones = []

  all_chords_good = True

  bad_chords = []

  for c in cscore:
    tones = sorted(list(set([t[4] % 12 for t in c])))
    chords_tones.append(tones)
    all_tones.extend(tones)

    if tones not in ALL_CHORDS:
      all_chords_good = False
      bad_chords.append(tones)

  analysis['Number of notes'] = len(score_notes)
  analysis['Number of chords'] = len(cscore)
  analysis['Score tones'] = sorted(list(set(all_tones)))
  analysis['Shortest chord'] = sorted(min(chords_tones, key=len))
  analysis['Longest chord'] = sorted(max(chords_tones, key=len))
  analysis['All chords good'] = all_chords_good
  analysis['Bad chords'] = bad_chords

  return analysis

#===============================================================================

def find_closest_tone(tones, tone):
  return min(tones, key=lambda x:abs(x-tone))

#===============================================================================

def flip_enhanced_score_notes(enhanced_score_notes):

    min_pitch = min([e[4] for e in enhanced_score_notes if e[3] != 9])

    fliped_score_pitches = [127 - e[4]for e in enhanced_score_notes if e[3] != 9]

    delta_min_pitch = min_pitch - min([p for p in fliped_score_pitches])

    output_score = copy.deepcopy(enhanced_score_notes)

    for e in output_score:
        if e[3] != 9:
            e[4] = (127 - e[4]) + delta_min_pitch

    return output_score

#===============================================================================

def patch_to_instrument_family(MIDI_patch, drums_patch=128):

  if 0 <= MIDI_patch < 128:
    return MIDI_patch // 8, MIDI_Instruments_Families[MIDI_patch // 8]

  elif MIDI_patch == drums_patch:
    return MIDI_patch // 8, MIDI_Instruments_Families[16]

  else:
    return -1, MIDI_Instruments_Families[-1]

#===============================================================================

def patch_list_from_enhanced_score_notes(enhanced_score_notes, 
                                         default_patch=0, 
                                         drums_patch=9,
                                         verbose=False
                                         ):

  patches = [-1] * 16

  for idx, e in enumerate(enhanced_score_notes):
    if e[0] == 'note':
      if e[3] != 9:
          if patches[e[3]] == -1:
              patches[e[3]] = e[6]
          else:
              if patches[e[3]] != e[6]:
                if e[6] in patches:
                  e[3] = patches.index(e[6])
                else:
                  if -1 in patches:
                      patches[patches.index(-1)] = e[6]
                  else:
                    patches[-1] = e[6]

                    if verbose:
                      print('=' * 70)
                      print('WARNING! Composition has more than 15 patches!')
                      print('Conflict note number:', idx)
                      print('Conflict channel number:', e[3])
                      print('Conflict patch number:', e[6])

  patches = [p if p != -1 else default_patch for p in patches]

  patches[9] = drums_patch

  if verbose:
    print('=' * 70)
    print('Composition patches')
    print('=' * 70)
    for c, p in enumerate(patches):
      print('Cha', str(c).zfill(2), '---', str(p).zfill(3), Number2patch[p])
    print('=' * 70)

  return patches

#===============================================================================

def patch_enhanced_score_notes(enhanced_score_notes, 
                                default_patch=0, 
                                drums_patch=9,
                                verbose=False
                                ):
  
    #===========================================================================    
  
    enhanced_score_notes_with_patch_changes = []

    patches = [-1] * 16

    overflow_idx = -1

    for idx, e in enumerate(enhanced_score_notes):
      if e[0] == 'note':
        if e[3] != 9:
            if patches[e[3]] == -1:
                patches[e[3]] = e[6]
            else:
                if patches[e[3]] != e[6]:
                  if e[6] in patches:
                    e[3] = patches.index(e[6])
                  else:
                    if -1 in patches:
                        patches[patches.index(-1)] = e[6]
                    else:
                        overflow_idx = idx
                        break

      enhanced_score_notes_with_patch_changes.append(e)

    #===========================================================================

    overflow_patches = []

    if overflow_idx != -1:
      for idx, e in enumerate(enhanced_score_notes[overflow_idx:]):
        if e[0] == 'note':
          if e[3] != 9:
            if e[6] not in patches:
              if e[6] not in overflow_patches:
                overflow_patches.append(e[6])
                enhanced_score_notes_with_patch_changes.append(['patch_change', e[1], e[3], e[6]])
            else:
              e[3] = patches.index(e[6])

          enhanced_score_notes_with_patch_changes.append(e)

    #===========================================================================

    patches = [p if p != -1 else default_patch for p in patches]

    patches[9] = drums_patch

    #===========================================================================

    if verbose:
      print('=' * 70)
      print('Composition patches')
      print('=' * 70)
      for c, p in enumerate(patches):
        print('Cha', str(c).zfill(2), '---', str(p).zfill(3), Number2patch[p])
      print('=' * 70)

      if overflow_patches:
        print('Extra composition patches')
        print('=' * 70)
        for c, p in enumerate(overflow_patches):
          print(str(p).zfill(3), Number2patch[p])
        print('=' * 70)

    return enhanced_score_notes_with_patch_changes, patches, overflow_patches

#===============================================================================

def pitches_to_tones(pitches):
  return [p % 12 for p in pitches]

#===============================================================================

def tones_to_pitches(tones, base_octave=5):
  return [(base_octave * 12) + t for t in tones]

#===============================================================================

def transpose_tones(tones, transpose_value=0):
  return [((60+t)+transpose_value) % 12 for t in tones]

#===============================================================================

def transpose_pitches(pitches, transpose_value=0):
  return [max(1, min(127, p+transpose_value)) for p in pitches]

#===============================================================================

def reverse_enhanced_score_notes(escore_notes):

  score = recalculate_score_timings(escore_notes)

  ematrix = escore_notes_to_escore_matrix(score, reverse_matrix=True)
  e_score = escore_matrix_to_original_escore_notes(ematrix)

  reversed_score = recalculate_score_timings(e_score)

  return reversed_score

#===============================================================================

def delta_pitches(escore_notes, pitches_index=4):

  pitches = [p[pitches_index] for p in escore_notes]
  
  return [a-b for a, b in zip(pitches[:-1], pitches[1:])]

#===============================================================================

def even_timings(escore_notes, 
                 times_idx=1, 
                 durs_idx=2
                 ):

  esn = copy.deepcopy(escore_notes)

  for e in esn:

    if e[times_idx] != 0:
      if e[times_idx] % 2 != 0:
        e[times_idx] += 1

    if e[durs_idx] % 2 != 0:
      e[durs_idx] += 1

  return esn

#===============================================================================

def find_next_bar(escore_notes, bar_time, start_note_idx, cur_bar):
  for e in escore_notes[start_note_idx:]:
    if e[1] // bar_time > cur_bar:
      return e, escore_notes.index(e)

#===============================================================================

def align_escore_notes_to_bars(escore_notes,
                               bar_time=4000,
                               trim_durations=False,
                               split_durations=False
                               ):

  #=============================================================================

  aligned_escore_notes = copy.deepcopy(escore_notes)

  abs_time = 0
  nidx = 0
  delta = 0
  bcount = 0
  next_bar = [0]

  #=============================================================================

  while next_bar:

    next_bar = find_next_bar(escore_notes, bar_time, nidx, bcount)

    if next_bar:

      gescore_notes = escore_notes[nidx:next_bar[1]]
    else:
      gescore_notes = escore_notes[nidx:]

    original_timings = [delta] + [(b[1]-a[1]) for a, b in zip(gescore_notes[:-1], gescore_notes[1:])]
    adj_timings = adjust_numbers_to_sum(original_timings, bar_time)

    for t in adj_timings:

      abs_time += t

      aligned_escore_notes[nidx][1] = abs_time
      aligned_escore_notes[nidx][2] -= int(bar_time // 200)

      nidx += 1

    if next_bar:
      delta = escore_notes[next_bar[1]][1]-escore_notes[next_bar[1]-1][1]
    bcount += 1

  #=============================================================================

  aligned_adjusted_escore_notes = []
  bcount = 0

  for a in aligned_escore_notes:
    bcount = a[1] // bar_time
    nbtime = bar_time * (bcount+1)

    if a[1]+a[2] > nbtime and a[3] != 9:
      if trim_durations or split_durations:
        ddiff = ((a[1]+a[2])-nbtime)
        aa = copy.deepcopy(a)
        aa[2] = a[2] - ddiff
        aligned_adjusted_escore_notes.append(aa)

        if split_durations:
          aaa = copy.deepcopy(a)
          aaa[1] = a[1]+aa[2]
          aaa[2] = ddiff

          aligned_adjusted_escore_notes.append(aaa)

      else:
        aligned_adjusted_escore_notes.append(a)

    else:
      aligned_adjusted_escore_notes.append(a)

  #=============================================================================

  return aligned_adjusted_escore_notes

#===============================================================================

def escore_notes_averages(escore_notes, 
                          times_index=1, 
                          durs_index=2,
                          chans_index=3, 
                          ptcs_index=4, 
                          vels_index=5,
                          average_drums=False,
                          score_is_delta=False,
                          return_ptcs_and_vels=False
                          ):
  
  if score_is_delta:
    if average_drums:
      times = [e[times_index] for e in escore_notes if e[times_index] != 0]
    else:
      times = [e[times_index] for e in escore_notes if e[times_index] != 0 and e[chans_index] != 9]

  else:
    descore_notes = delta_score_notes(escore_notes)
    if average_drums:
      times = [e[times_index] for e in descore_notes if e[times_index] != 0]
    else:
      times = [e[times_index] for e in descore_notes if e[times_index] != 0 and e[chans_index] != 9]
      
  if average_drums:
    durs = [e[durs_index] for e in escore_notes]
  else:
    durs = [e[durs_index] for e in escore_notes if e[chans_index] != 9]

  if return_ptcs_and_vels:
    if average_drums:
      ptcs = [e[ptcs_index] for e in escore_notes]
      vels = [e[vels_index] for e in escore_notes]
    else:
      ptcs = [e[ptcs_index] for e in escore_notes if e[chans_index] != 9]
      vels = [e[vels_index] for e in escore_notes if e[chans_index] != 9]      

    return [sum(times) / len(times), sum(durs) / len(durs), sum(ptcs) / len(ptcs), sum(vels) / len(vels)]
  
  else:
    return [sum(times) / len(times), sum(durs) / len(durs)]

#===============================================================================

def adjust_escore_notes_timings(escore_notes, 
                                adj_k=1, 
                                times_index=1, 
                                durs_index=2, 
                                score_is_delta=False, 
                                return_delta_scpre=False
                                ):

  if score_is_delta:
    adj_escore_notes = copy.deepcopy(escore_notes)
  else:
    adj_escore_notes = delta_score_notes(escore_notes)

  for e in adj_escore_notes:

    if e[times_index] != 0:
      e[times_index] = max(1, round(e[times_index] * adj_k))

    e[durs_index] = max(1, round(e[durs_index] * adj_k))

  if return_delta_scpre:
    return adj_escore_notes

  else:
    return delta_score_to_abs_score(adj_escore_notes)

#===============================================================================

def escore_notes_delta_times(escore_notes,
                             times_index=1
                             ):

  descore_notes = delta_score_notes(escore_notes)

  return [e[times_index] for e in descore_notes]

#===============================================================================

def escore_notes_durations(escore_notes,
                            durs_index=1
                            ):

  descore_notes = delta_score_notes(escore_notes)

  return [e[durs_index] for e in descore_notes]

#===============================================================================

def transpose_escore_notes(escore_notes, 
                            transpose_value=0, 
                            channel_index=3, 
                            pitches_index=4
                            ):

  tr_escore_notes = copy.deepcopy(escore_notes)

  for e in tr_escore_notes:
    if e[channel_index] != 9:
      e[pitches_index] = max(1, min(127, e[pitches_index] + transpose_value))

  return tr_escore_notes

#===============================================================================

def transpose_escore_notes_to_pitch(escore_notes, 
                                    target_pitch_value=60, 
                                    channel_index=3, 
                                    pitches_index=4
                                    ):

  tr_escore_notes = copy.deepcopy(escore_notes)

  transpose_delta = int(round(target_pitch_value)) - int(round(escore_notes_averages(escore_notes, return_ptcs_and_vels=True)[2]))

  for e in tr_escore_notes:
    if e[channel_index] != 9:
      e[pitches_index] = max(1, min(127, e[pitches_index] + transpose_delta))

  return tr_escore_notes

#===============================================================================

def tone_type(tone, 
              return_tone_type_index=True
              ):

  tone = tone % 12

  if tone in BLACK_NOTES:
    if return_tone_type_index:
      return CHORDS_TYPES.index('BLACK')
    else:
      return "BLACK"

  else:
    if return_tone_type_index:
      return CHORDS_TYPES.index('WHITE')
    else:
      return "WHITE"

#===============================================================================

def adjust_escore_notes_to_average(escore_notes,
                                   trg_avg,
                                   min_value=1,
                                   max_value=4000,
                                   times_index=1,
                                   durs_index=2,
                                   score_is_delta=False,
                                   return_delta_scpre=False
                                   ):
    if score_is_delta:
      delta_escore_notes = copy.deepcopy(escore_notes)

    else:
      delta_escore_notes = delta_score_notes(escore_notes)

    times = [[e[times_index], e[durs_index]] for e in delta_escore_notes]

    filtered_values = [value for value in times if min_value <= value[0] <= max_value]

    if not filtered_values:
        return escore_notes

    current_avg = sum([v[0] for v in filtered_values]) / len([v[0] for v in filtered_values])
    scale_factor = trg_avg / current_avg

    adjusted_values = [[value[0] * scale_factor, value[1] * scale_factor] for value in filtered_values]

    total_difference = trg_avg * len([v[0] for v in filtered_values]) - sum([v[0] for v in adjusted_values])
    adjustment_per_value = total_difference / len(filtered_values)

    final_values = [[value[0] + adjustment_per_value, value[1] + adjustment_per_value] for value in adjusted_values]

    while abs(sum([v[0] for v in final_values]) / len(final_values) - trg_avg) > 1e-6:
        total_difference = trg_avg * len(final_values) - sum([v[0] for v in final_values])
        adjustment_per_value = total_difference / len(final_values)
        final_values = [[value[0] + adjustment_per_value, value[1] + adjustment_per_value] for value in final_values]

    final_values = [[round(value[0]), round(value[1])] for value in final_values]

    adjusted_delta_score = copy.deepcopy(delta_escore_notes)

    j = 0

    for i in range(len(adjusted_delta_score)):
        if min_value <= adjusted_delta_score[i][1] <= max_value:
            adjusted_delta_score[i][times_index] = final_values[j][0]
            adjusted_delta_score[i][durs_index] = final_values[j][1]
            j += 1

    adjusted_escore_notes = delta_score_to_abs_score(adjusted_delta_score)

    if return_delta_scpre:
      return adjusted_delta_score

    else:
      return adjusted_escore_notes

#===============================================================================

def calculate_combined_distances(array_of_arrays,
                                  combine_hamming_distance=True,
                                  combine_jaccard_similarity=True, 
                                  combine_pearson_correlation=True,
                                  pad_value=None
                                  ):

  binary_arrays = array_of_arrays
  binary_array_len = len(binary_arrays)

  hamming_distances = [[0] * binary_array_len for _ in range(binary_array_len)]
  jaccard_similarities = [[0] * binary_array_len for _ in range(binary_array_len)]
  pearson_correlations = [[0] * binary_array_len for _ in range(binary_array_len)]

  for i in range(binary_array_len):
      for j in range(i + 1, binary_array_len):
          hamming_distances[i][j] = hamming_distance(binary_arrays[i], binary_arrays[j], pad_value)
          hamming_distances[j][i] = hamming_distances[i][j]
          
          jaccard_similarities[i][j] = jaccard_similarity(binary_arrays[i], binary_arrays[j], pad_value)
          jaccard_similarities[j][i] = jaccard_similarities[i][j]
          
          pearson_correlations[i][j] = pearson_correlation(binary_arrays[i], binary_arrays[j], pad_value)
          pearson_correlations[j][i] = pearson_correlations[i][j]

  max_hamming = max(max(row) for row in hamming_distances)
  min_hamming = min(min(row) for row in hamming_distances)
  normalized_hamming = [[(val - min_hamming) / (max_hamming - min_hamming) for val in row] for row in hamming_distances]

  max_jaccard = max(max(row) for row in jaccard_similarities)
  min_jaccard = min(min(row) for row in jaccard_similarities)
  normalized_jaccard = [[(val - min_jaccard) / (max_jaccard - min_jaccard) for val in row] for row in jaccard_similarities]

  max_pearson = max(max(row) for row in pearson_correlations)
  min_pearson = min(min(row) for row in pearson_correlations)
  normalized_pearson = [[(val - min_pearson) / (max_pearson - min_pearson) for val in row] for row in pearson_correlations]

  selected_metrics = 0

  if combine_hamming_distance:
    selected_metrics += normalized_hamming[i][j]
  
  if combine_jaccard_similarity:
    selected_metrics += (1 - normalized_jaccard[i][j])

  if combine_pearson_correlation:
    selected_metrics += (1 - normalized_pearson[i][j])

  combined_metric = [[selected_metrics for i in range(binary_array_len)] for j in range(binary_array_len)]

  return combined_metric

#===============================================================================

def solo_piano_escore_notes(escore_notes,
                            channels_index=3,
                            pitches_index=4,
                            patches_index=6,
                            keep_drums=False,
                            ):

  cscore = chordify_score([1000, escore_notes])

  sp_escore_notes = []

  for c in cscore:

    seen = []
    chord = []

    for cc in c:
      if cc[pitches_index] not in seen:

          if cc[channels_index] != 9:
            cc[channels_index] = 0
            cc[patches_index] = 0
            
            chord.append(cc)
            seen.append(cc[pitches_index])
          
          else:
            if keep_drums:
              chord.append(cc)
              seen.append(cc[pitches_index])

    sp_escore_notes.append(chord)

  return flatten(sp_escore_notes)

#===============================================================================

def strip_drums_from_escore_notes(escore_notes, 
                                  channels_index=3
                                  ):
  
  return [e for e in escore_notes if e[channels_index] != 9]

#===============================================================================

def fixed_escore_notes_timings(escore_notes,
                               fixed_durations=False,
                               fixed_timings_multiplier=1,
                               custom_fixed_time=-1,
                               custom_fixed_dur=-1
                               ):

  fixed_timings_escore_notes = delta_score_notes(escore_notes, even_timings=True)

  mode_time = round(Counter([e[1] for e in fixed_timings_escore_notes if e[1] != 0]).most_common()[0][0] * fixed_timings_multiplier)

  if mode_time % 2 != 0:
    mode_time += 1

  mode_dur = round(Counter([e[2] for e in fixed_timings_escore_notes if e[2] != 0]).most_common()[0][0] * fixed_timings_multiplier)

  if mode_dur % 2 != 0:
    mode_dur += 1

  for e in fixed_timings_escore_notes:
    if e[1] != 0:
      
      if custom_fixed_time > 0:
        e[1] = custom_fixed_time
      
      else:
        e[1] = mode_time

    if fixed_durations:
      
      if custom_fixed_dur > 0:
        e[2] = custom_fixed_dur
      
      else:
        e[2] = mode_dur

  return delta_score_to_abs_score(fixed_timings_escore_notes)

#===============================================================================

def summarize_escore_notes(escore_notes, 
                           summary_length_in_chords=128, 
                           preserve_timings=True,
                           preserve_durations=False,
                           time_threshold=12,
                           min_sum_chord_len=2,
                           use_tones_chords=True
                           ):

    cscore = chordify_score([d[1:] for d in delta_score_notes(escore_notes)])

    summary_length_in_chords = min(len(cscore), summary_length_in_chords)

    ltthresh = time_threshold // 2
    uttresh = time_threshold * 2

    mc_time = Counter([c[0][0] for c in cscore if c[0][2] != 9 and ltthresh < c[0][0] < uttresh]).most_common()[0][0]

    pchords = []

    for c in cscore:
      if use_tones_chords:
        pchords.append([c[0][0]] + pitches_to_tones_chord(chord_to_pchord(c)))
        
      else:
        pchords.append([c[0][0]] + chord_to_pchord(c))

    step = round(len(pchords) / summary_length_in_chords)

    samples = []

    for i in range(0, len(pchords), step):
      samples.append(tuple([tuple(d) for d in pchords[i:i+step]]))

    summarized_escore_notes = []

    for i, s in enumerate(samples):

      best_chord = list([v[0] for v in Counter(s).most_common() if v[0][0] == mc_time and len(v[0]) > min_sum_chord_len])

      if not best_chord:
        best_chord = list([v[0] for v in Counter(s).most_common() if len(v[0]) > min_sum_chord_len])
        
        if not best_chord:
          best_chord = list([Counter(s).most_common()[0][0]])

      chord = copy.deepcopy(cscore[[ss for ss in s].index(best_chord[0])+(i*step)])

      if preserve_timings:

        if not preserve_durations:

          if i > 0:

            pchord = summarized_escore_notes[-1]

            for pc in pchord:
              pc[1] = min(pc[1], chord[0][0])

      else:

        chord[0][0] = 1

        for c in chord:
          c[1] = 1  

      summarized_escore_notes.append(chord)

    summarized_escore_notes = summarized_escore_notes[:summary_length_in_chords]

    return [['note'] + d for d in delta_score_to_abs_score(flatten(summarized_escore_notes), times_idx=0)]

#===============================================================================

def compress_patches_in_escore_notes(escore_notes,
                                     num_patches=4,
                                     group_patches=False
                                     ):

  if num_patches > 4:
    n_patches = 4
  elif num_patches < 1:
    n_patches = 1
  else:
    n_patches = num_patches

  if group_patches:
    patches_set = sorted(set([e[6] for e in escore_notes]))
    trg_patch_list = []
    seen = []
    for p in patches_set:
      if p // 8 not in seen:
        trg_patch_list.append(p)
        seen.append(p // 8)

    trg_patch_list = sorted(trg_patch_list)

  else:
    trg_patch_list = sorted(set([e[6] for e in escore_notes]))

  if 128 in trg_patch_list and n_patches > 1:
    trg_patch_list = trg_patch_list[:n_patches-1] + [128]
  else:
    trg_patch_list = trg_patch_list[:n_patches]

  new_escore_notes = []

  for e in escore_notes:
    if e[6] in trg_patch_list:
      new_escore_notes.append(e)

  return new_escore_notes

#===============================================================================

def compress_patches_in_escore_notes_chords(escore_notes,
                                            max_num_patches_per_chord=4,
                                            group_patches=True,
                                            root_grouped_patches=False
                                            ):

  if max_num_patches_per_chord > 4:
    n_patches = 4
  elif max_num_patches_per_chord < 1:
    n_patches = 1
  else:
    n_patches = max_num_patches_per_chord

  cscore = chordify_score([1000, sorted(escore_notes, key=lambda x: (x[1], x[6]))])

  new_escore_notes = []

  for c in cscore:

    if group_patches:
      patches_set = sorted(set([e[6] for e in c]))
      trg_patch_list = []
      seen = []
      for p in patches_set:
        if p // 8 not in seen:
          trg_patch_list.append(p)
          seen.append(p // 8)

      trg_patch_list = sorted(trg_patch_list)

    else:
      trg_patch_list = sorted(set([e[6] for e in c]))

    if 128 in trg_patch_list and n_patches > 1:
      trg_patch_list = trg_patch_list[:n_patches-1] + [128]
    else:
      trg_patch_list = trg_patch_list[:n_patches]

    for ccc in c:

      cc = copy.deepcopy(ccc)

      if group_patches:
        if cc[6] // 8 in [t // 8 for t in trg_patch_list]:
          if root_grouped_patches:
            cc[6] = (cc[6] // 8) * 8
          new_escore_notes.append(cc)

      else:
        if cc[6] in trg_patch_list:
          new_escore_notes.append(cc)

  return new_escore_notes

#===============================================================================

def escore_notes_delta_times(escore_notes, 
                             timings_index=1, 
                             channels_index=3, 
                             omit_zeros=False, 
                             omit_drums=False
                            ):

  if omit_drums:

    score = [e for e in escore_notes if e[channels_index] != 9]
    dtimes = [score[0][timings_index]] + [b[timings_index]-a[timings_index] for a, b in zip(score[:-1], score[1:])]

  else:
    dtimes = [escore_notes[0][timings_index]] + [b[timings_index]-a[timings_index] for a, b in zip(escore_notes[:-1], escore_notes[1:])]
  
  if omit_zeros:
    dtimes = [d for d in dtimes if d != 0]
  
  return dtimes

#===============================================================================

def monophonic_check(escore_notes, times_index=1):
  return len(escore_notes) == len(set([e[times_index] for e in escore_notes]))

#===============================================================================

def count_escore_notes_patches(escore_notes, patches_index=6):
  return [list(c) for c in Counter([e[patches_index] for e in escore_notes]).most_common()]

#===============================================================================

def escore_notes_medley(list_of_escore_notes, 
                        list_of_labels=None,
                        pause_time_value=255
                        ):

  if list_of_labels is not None:
    labels = [str(l) for l in list_of_labels] + ['No label'] * (len(list_of_escore_notes)-len(list_of_labels))

  medley = []

  time = 0

  for i, m in enumerate(list_of_escore_notes):

    if list_of_labels is not None:
      medley.append(['text_event', time, labels[i]])

    pe = m[0]

    for mm in m:

      time += mm[1] - pe[1]

      mmm = copy.deepcopy(mm)
      mmm[1] = time

      medley.append(mmm)

      pe = mm

    time += pause_time_value

  return medley

#===============================================================================

def smooth_escore_notes(escore_notes):

  values = [e[4] % 24 for e in escore_notes]

  smoothed = [values[0]]

  for i in range(1, len(values)):
      if abs(smoothed[-1] - values[i]) >= 12:
          if smoothed[-1] < values[i]:
              smoothed.append(values[i] - 12)
          else:
              smoothed.append(values[i] + 12)
      else:
          smoothed.append(values[i])

  smoothed_score = copy.deepcopy(escore_notes)

  for i, e in enumerate(smoothed_score):
    esn_octave = escore_notes[i][4] // 12
    e[4] = (esn_octave * 12) + smoothed[i]

  return smoothed_score

#===============================================================================

def add_base_to_escore_notes(escore_notes,
                             base_octave=2, 
                             base_channel=2, 
                             base_patch=35, 
                             base_max_velocity=120,
                             return_base=False
                             ):
  

  score = copy.deepcopy(escore_notes)

  cscore = chordify_score([1000, score])

  base_score = []

  for c in cscore:
    chord = sorted([e for e in c if e[3] != 9], key=lambda x: x[4], reverse=True)
    base_score.append(chord[-1])

  base_score = smooth_escore_notes(base_score)

  for e in base_score:
    e[3] = base_channel
    e[4] = (base_octave * 12) + (e[4] % 12)
    e[5] = e[4]
    e[6] = base_patch

  adjust_score_velocities(base_score, base_max_velocity)

  if return_base:
    final_score = sorted(base_score, key=lambda x: (x[1], -x[4], x[6]))

  else:
    final_score = sorted(escore_notes + base_score, key=lambda x: (x[1], -x[4], x[6]))

  return final_score

#===============================================================================

def add_drums_to_escore_notes(escore_notes, 
                              heavy_drums_pitches=[36, 38, 47],
                              heavy_drums_velocity=110,
                              light_drums_pitches=[51, 54],
                              light_drums_velocity=127,
                              drums_max_velocity=127,
                              drums_ratio_time_divider=4,
                              return_drums=False
                              ):

  score = copy.deepcopy([e for e in escore_notes if e[3] != 9])

  cscore = chordify_score([1000, score])

  drums_score = []

  for c in cscore:
    min_dur = max(1, min([e[2] for e in c]))
    if not (c[0][1] % drums_ratio_time_divider):
      drum_note = ['note', c[0][1], min_dur, 9, heavy_drums_pitches[c[0][4] % len(heavy_drums_pitches)], heavy_drums_velocity, 128]
    else:
      drum_note = ['note', c[0][1], min_dur, 9, light_drums_pitches[c[0][4] % len(light_drums_pitches)], light_drums_velocity, 128]
    drums_score.append(drum_note)

  adjust_score_velocities(drums_score, drums_max_velocity)

  if return_drums:
    final_score = sorted(drums_score, key=lambda x: (x[1], -x[4], x[6]))

  else:
    final_score = sorted(score + drums_score, key=lambda x: (x[1], -x[4], x[6]))

  return final_score

#===============================================================================

def escore_notes_lrno_pattern(escore_notes, mode='chords'):

  cscore = chordify_score([1000, escore_notes])

  checked_cscore = advanced_check_and_fix_chords_in_chordified_score(cscore)

  chords_toks = []
  chords_idxs = []

  for i, c in enumerate(checked_cscore[0]):

    pitches = sorted([p[4] for p in c if p[3] != 9], reverse=True)
    tchord = pitches_to_tones_chord(pitches)

    if tchord:
      
      if mode == 'chords':
        token = ALL_CHORDS_FULL.index(tchord)
      
      elif mode == 'high pitches':
        token = pitches[0]

      elif mode == 'high pitches tones':
        token = pitches[0] % 12

      else:
        token = ALL_CHORDS_FULL.index(tchord)

      chords_toks.append(token)
      chords_idxs.append(i)

  lrno_pats = find_lrno_patterns(chords_toks)

  if lrno_pats:

    lrno_pattern = list(lrno_pats[0][2])

    start_idx = chords_idxs[find_pattern_start_indexes(chords_toks, lrno_pattern)[0]]
    end_idx = chords_idxs[start_idx + len(lrno_pattern)]

    return recalculate_score_timings(flatten(cscore[start_idx:end_idx]))

  else:
    return None

#===============================================================================

def escore_notes_times_tones(escore_notes, 
                             tones_mode='dominant', 
                             return_abs_times=True,
                             omit_drums=False
                             ):

  cscore = chordify_score([1000, escore_notes])
  
  tones = chordified_score_pitches(cscore, return_tones=True, mode=tones_mode, omit_drums=omit_drums)

  if return_abs_times:
    times = sorted([c[0][1] for c in cscore])
  
  else:
    times = escore_notes_delta_times(escore_notes, omit_zeros=True, omit_drums=omit_drums)
    
    if len(times) != len(tones):
      times = [0] + times

  return [[t, to] for t, to in zip(times, tones)]

#===============================================================================

def escore_notes_middle(escore_notes, 
                        length=10, 
                        use_chords=True
                        ):

  if use_chords:
    score = chordify_score([1000, escore_notes])

  else:
    score = escore_notes

  middle_idx = len(score) // 2

  slen = min(len(score) // 2, length // 2)

  start_idx = middle_idx - slen
  end_idx = middle_idx + slen

  if use_chords:
    return flatten(score[start_idx:end_idx])

  else:
    return score[start_idx:end_idx]

#===============================================================================

def escore_notes_to_parsons_code(escore_notes,
                                 times_index=1,
                                 pitches_index=4,
                                 return_as_list=False
                                 ):
  
  parsons = "*"
  parsons_list = []

  prev = ['note', -1, -1, -1, -1, -1, -1]

  for e in escore_notes:
    if e[times_index] != prev[times_index]:

      if e[pitches_index] > prev[pitches_index]:
          parsons += "U"
          parsons_list.append(1)

      elif e[pitches_index] < prev[pitches_index]:
          parsons += "D"
          parsons_list.append(-1)

      elif e[pitches_index] == prev[pitches_index]:
          parsons += "R"
          parsons_list.append(0)
      
      prev = e

  if return_as_list:
    return parsons_list
  
  else:
    return parsons

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]

#=======================================================================================================
# This is the end of misc module
#=======================================================================================================
