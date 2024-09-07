#=======================================================================================================
#
# Tegridy MIDI Python Module
#
# Version 24.9.6
#
# Project Los Angeles
# Tegridy Code 2024
#
#=======================================================================================================

import copy
import statistics

from tegridymidi.processors import recalculate_score_timings, delta_score_notes, delta_score_to_abs_score
from tegridymidi.chords import chordify_score
from tegridymidi.constants import Number2patch, MIDI_Instruments_Families, ALL_CHORDS
from tegridymidi.helpers import adjust_numbers_to_sum
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

__all__ = [name for name in globals() if not name.startswith('_')]

#=======================================================================================================
# This is the end of tegridymidi module
#=======================================================================================================
