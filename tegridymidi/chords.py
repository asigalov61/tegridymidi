#=======================================================================================================
# Tegridy MIDI Chords module
#=======================================================================================================

import copy
import random
import statistics
from collections import Counter
from itertools import combinations, groupby

from tegridymidi.processors import chordify_score
from tegridymidi.helpers import find_exact_match_variable_length, lists_intersections, lists_sym_differences
from tegridymidi.constants import ALL_CHORDS, ALL_CHORDS_FULL, ALL_CHORDS_FILTERED, ALL_CHORDS_SORTED
from tegridymidi.constants import WHITE_NOTES, BLACK_NOTES, CHORDS_TYPES

#===============================================================================

def bad_chord(chord):
    bad = any(b - a == 1 for a, b in zip(chord, chord[1:]))
    if (0 in chord) and (11 in chord):
      bad = True
    
    return bad

#===============================================================================

def find_closest_tone(tones, tone):
  return min(tones, key=lambda x:abs(x-tone))

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

def validate_pitches_chord(pitches_chord, return_sorted = True):

    pitches_chord = sorted(list(set([x for x in pitches_chord if 0 < x < 128])))

    tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))

    if not bad_chord(tones_chord):
      if return_sorted:
        pitches_chord.sort(reverse=True)
      return pitches_chord
    
    else:
      if 0 in tones_chord and 11 in tones_chord:
        tones_chord.remove(0)

      fixed_tones = [[a, b] for a, b in zip(tones_chord, tones_chord[1:]) if b-a != 1]

      fixed_tones_chord = []
      for f in fixed_tones:
        fixed_tones_chord.extend(f)
      fixed_tones_chord = list(set(fixed_tones_chord))
      
      fixed_pitches_chord = []

      for p in pitches_chord:
        if (p % 12) in fixed_tones_chord:
          fixed_pitches_chord.append(p)

      if return_sorted:
        fixed_pitches_chord.sort(reverse=True)

    return fixed_pitches_chord

#===============================================================================

def validate_chord_pitches(chord, channel_to_check = 0, return_sorted = True):

    pitches_chord = sorted(list(set([x[4] for x in chord if 0 < x[4] < 128 and x[3] == channel_to_check])))

    if pitches_chord:

      tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))

      if not bad_chord(tones_chord):
        if return_sorted:
          chord.sort(key = lambda x: x[4], reverse=True)
        return chord
      
      else:
        if 0 in tones_chord and 11 in tones_chord:
          tones_chord.remove(0)

        fixed_tones = [[a, b] for a, b in zip(tones_chord, tones_chord[1:]) if b-a != 1]

        fixed_tones_chord = []
        for f in fixed_tones:
          fixed_tones_chord.extend(f)
        fixed_tones_chord = list(set(fixed_tones_chord))
        
        fixed_chord = []

        for c in chord:
          if c[3] == channel_to_check:
            if (c[4] % 12) in fixed_tones_chord:
              fixed_chord.append(c)
          else:
            fixed_chord.append(c)

        if return_sorted:
          fixed_chord.sort(key = lambda x: x[4], reverse=True)
      
        return fixed_chord 

    else:
      chord.sort(key = lambda x: x[4], reverse=True)
      return chord

#===============================================================================

def advanced_validate_chord_pitches(chord, channel_to_check = 0, return_sorted = True):

    pitches_chord = sorted(list(set([x[4] for x in chord if 0 < x[4] < 128 and x[3] == channel_to_check])))

    if pitches_chord:

      tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))

      if not bad_chord(tones_chord):
        if return_sorted:
          chord.sort(key = lambda x: x[4], reverse=True)
        return chord

      else:
        bad_chord_indices = list(set([i for s in [[tones_chord.index(a), tones_chord.index(b)] for a, b in zip(tones_chord, tones_chord[1:]) if b-a == 1] for i in s]))
        
        good_tones_chord = find_exact_match_variable_length(ALL_CHORDS, tones_chord, bad_chord_indices)
        
        if good_tones_chord is not None:
        
          fixed_chord = []

          for c in chord:
            if c[3] == channel_to_check:
              if (c[4] % 12) in good_tones_chord:
                fixed_chord.append(c)
            else:
              fixed_chord.append(c)

          if return_sorted:
            fixed_chord.sort(key = lambda x: x[4], reverse=True)

        else:

          if 0 in tones_chord and 11 in tones_chord:
            tones_chord.remove(0)

          fixed_tones = [[a, b] for a, b in zip(tones_chord, tones_chord[1:]) if b-a != 1]

          fixed_tones_chord = []
          for f in fixed_tones:
            fixed_tones_chord.extend(f)
          fixed_tones_chord = list(set(fixed_tones_chord))
          
          fixed_chord = []

          for c in chord:
            if c[3] == channel_to_check:
              if (c[4] % 12) in fixed_tones_chord:
                fixed_chord.append(c)
            else:
              fixed_chord.append(c)

          if return_sorted:
            fixed_chord.sort(key = lambda x: x[4], reverse=True)     
      
      return fixed_chord 

    else:
      chord.sort(key = lambda x: x[4], reverse=True)
      return chord

#===============================================================================

def pitches_to_tones_chord(pitches):
  return sorted(set([p % 12 for p in pitches]))

#===============================================================================

def tones_chord_to_pitches(tones_chord, base_pitch=60):
  return [t+base_pitch for t in tones_chord if 0 <= t < 12]

#===============================================================================

def replace_bad_tones_chord(bad_tones_chord):
  bad_chord_p = [0] * 12
  for b in bad_tones_chord:
    bad_chord_p[b] = 1

  match_ratios = []
  good_chords = []
  for c in ALL_CHORDS:
    good_chord_p = [0] * 12
    for cc in c:
      good_chord_p[cc] = 1

    good_chords.append(good_chord_p)
    match_ratios.append(sum(i == j for i, j in zip(good_chord_p, bad_chord_p)) / len(good_chord_p))

  best_good_chord = good_chords[match_ratios.index(max(match_ratios))]

  replaced_chord = []
  for i in range(len(best_good_chord)):
    if best_good_chord[i] == 1:
     replaced_chord.append(i)

  return [replaced_chord, max(match_ratios)]

#===============================================================================

def check_and_fix_chord(chord, 
                        channel_index=3,
                        pitch_index=4
                        ):

    tones_chord = sorted(set([t[pitch_index] % 12 for t in chord if t[channel_index] != 9]))

    notes_events = [t for t in chord if t[channel_index] != 9]
    notes_events.sort(key=lambda x: x[pitch_index], reverse=True)

    drums_events = [t for t in chord if t[channel_index] == 9]

    checked_and_fixed_chord = []

    if tones_chord:
        
        new_tones_chord = advanced_check_and_fix_tones_chord(tones_chord, high_pitch=notes_events[0][pitch_index])

        if new_tones_chord != tones_chord:

          if len(notes_events) > 1:
              checked_and_fixed_chord.extend([notes_events[0]])
              for cc in notes_events[1:]:
                  if cc[channel_index] != 9:
                      if (cc[pitch_index] % 12) in new_tones_chord:
                          checked_and_fixed_chord.extend([cc])
              checked_and_fixed_chord.extend(drums_events)
          else:
              checked_and_fixed_chord.extend([notes_events[0]])
        else:
          checked_and_fixed_chord.extend(chord)
    else:
        checked_and_fixed_chord.extend(chord)

    checked_and_fixed_chord.sort(key=lambda x: x[pitch_index], reverse=True)

    return checked_and_fixed_chord

#===============================================================================

def find_similar_tones_chord(tones_chord, 
                             max_match_threshold=1, 
                             randomize_chords_matches=False, 
                             custom_chords_list=[]):
  chord_p = [0] * 12
  for b in tones_chord:
    chord_p[b] = 1

  match_ratios = []
  good_chords = []

  if custom_chords_list:
    CHORDS = copy.deepcopy([list(x) for x in set(tuple(t) for t in custom_chords_list)])
  else:
    CHORDS = copy.deepcopy(ALL_CHORDS)

  if randomize_chords_matches:
    random.shuffle(CHORDS)

  for c in CHORDS:
    good_chord_p = [0] * 12
    for cc in c:
      good_chord_p[cc] = 1

    good_chords.append(good_chord_p)
    match_ratio = sum(i == j for i, j in zip(good_chord_p, chord_p)) / len(good_chord_p)
    if match_ratio < max_match_threshold:
      match_ratios.append(match_ratio)
    else:
      match_ratios.append(0)

  best_good_chord = good_chords[match_ratios.index(max(match_ratios))]

  similar_chord = []
  for i in range(len(best_good_chord)):
    if best_good_chord[i] == 1:
     similar_chord.append(i)

  return [similar_chord, max(match_ratios)]

#===============================================================================

def generate_tones_chords_progression(number_of_chords_to_generate=100, 
                                      start_tones_chord=[], 
                                      custom_chords_list=[]):

  if start_tones_chord:
    start_chord = start_tones_chord
  else:
    start_chord = random.choice(ALL_CHORDS)

  chord = []

  chords_progression = [start_chord]

  for i in range(number_of_chords_to_generate):
    if not chord:
      chord = start_chord

    if custom_chords_list:
      chord = find_similar_tones_chord(chord, randomize_chords_matches=True, custom_chords_list=custom_chords_list)[0]
    else:
      chord = find_similar_tones_chord(chord, randomize_chords_matches=True)[0]
    
    chords_progression.append(chord)

  return chords_progression

#===============================================================================

def check_and_fix_tones_chord(tones_chord):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  for c in tones_chord_combs:
    if c in ALL_CHORDS_FULL:
      checked_tones_chord = c
      break

  return sorted(checked_tones_chord)

#===============================================================================

def advanced_check_and_fix_tones_chord(tones_chord, high_pitch=0):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  for c in tones_chord_combs:
    if c in ALL_CHORDS_FULL:
      tchord = c

  if 0 < high_pitch < 128 and len(tchord) == 1:
    tchord = [high_pitch % 12]

  return tchord

#===============================================================================

def check_and_fix_chords_in_chordified_score(chordified_score,
                                             channels_index=3,
                                             pitches_index=4
                                             ):
  fixed_chordified_score = []

  bad_chords_counter = 0

  for c in chordified_score:

    tones_chord = sorted(set([t[pitches_index] % 12 for t in c if t[channels_index] != 9]))

    if tones_chord:

        if tones_chord not in ALL_CHORDS_SORTED:
          bad_chords_counter += 1

        while tones_chord not in ALL_CHORDS_SORTED:
          tones_chord.pop(0)

    new_chord = []

    c.sort(key = lambda x: x[pitches_index], reverse=True)

    for e in c:
      if e[channels_index] != 9:
        if e[pitches_index] % 12 in tones_chord:
          new_chord.append(e)

      else:
        new_chord.append(e)

    fixed_chordified_score.append(new_chord)

  return fixed_chordified_score, bad_chords_counter

#===============================================================================

def advanced_check_and_fix_chords_in_chordified_score(chordified_score,
                                                      channels_index=3,
                                                      pitches_index=4,
                                                      patches_index=6,
                                                      use_filtered_chords=False,
                                                      use_full_chords=True,
                                                      remove_duplicate_pitches=True,
                                                      fix_bad_pitches=False,
                                                      skip_drums=False
                                                      ):
  fixed_chordified_score = []

  bad_chords_counter = 0
  duplicate_pitches_counter = 0

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  for c in chordified_score:

    chord = copy.deepcopy(c)

    if remove_duplicate_pitches:

      chord.sort(key = lambda x: x[pitches_index], reverse=True)

      seen = set()
      ddchord = []

      for cc in chord:
        if cc[channels_index] != 9:

          if tuple([cc[pitches_index], cc[patches_index]]) not in seen:
            ddchord.append(cc)
            seen.add(tuple([cc[pitches_index], cc[patches_index]]))
          else:
            duplicate_pitches_counter += 1
        
        else:
          ddchord.append(cc)
      
      chord = copy.deepcopy(ddchord)
      
    tones_chord = sorted(set([t[pitches_index] % 12 for t in chord if t[channels_index] != 9]))

    if tones_chord:

        if tones_chord not in CHORDS:
          
          pitches_chord = sorted(set([p[pitches_index] for p in c if p[channels_index] != 9]), reverse=True)
          
          if len(tones_chord) == 2:
            tones_counts = Counter([p % 12 for p in pitches_chord]).most_common()

            if tones_counts[0][1] > 1:
              tones_chord = [tones_counts[0][0]]
            
            elif tones_counts[1][1] > 1:
              tones_chord = [tones_counts[1][0]]
            
            else:
              tones_chord = [pitches_chord[0] % 12]

          else:
            tones_chord_combs = [list(comb) for i in range(len(tones_chord)-1, 0, -1) for comb in combinations(tones_chord, i)]

            for co in tones_chord_combs:
              if co in CHORDS:
                tones_chord = co
                break

          if len(tones_chord) == 1:
            tones_chord = [pitches_chord[0] % 12]
            
          bad_chords_counter += 1

    chord.sort(key = lambda x: x[pitches_index], reverse=True)

    new_chord = set()
    pipa = []

    for e in chord:
      if e[channels_index] != 9:
        if e[pitches_index] % 12 in tones_chord:
          new_chord.add(tuple(e))
          pipa.append([e[pitches_index], e[patches_index]])

        elif (e[pitches_index]+1) % 12 in tones_chord:
          e[pitches_index] += 1
          new_chord.add(tuple(e))
          pipa.append([e[pitches_index], e[patches_index]])

        elif (e[pitches_index]-1) % 12 in tones_chord:
          e[pitches_index] -= 1
          new_chord.add(tuple(e))
          pipa.append([e[pitches_index], e[patches_index]])

    if fix_bad_pitches:

      bad_chord = set()

      for e in chord:
        if e[channels_index] != 9:
          
          if e[pitches_index] % 12 not in tones_chord:
            bad_chord.add(tuple(e))
          
          elif (e[pitches_index]+1) % 12 not in tones_chord:
            bad_chord.add(tuple(e))
          
          elif (e[pitches_index]-1) % 12 not in tones_chord:
            bad_chord.add(tuple(e))
      
      for bc in bad_chord:

        bc = list(bc)

        tone = find_closest_tone(tones_chord, bc[pitches_index] % 12)

        new_pitch =  ((bc[pitches_index] // 12) * 12) + tone

        if [new_pitch, bc[patches_index]] not in pipa:
          bc[pitches_index] = new_pitch
          new_chord.add(tuple(bc))
          pipa.append([[new_pitch], bc[patches_index]])

    if not skip_drums:
      for e in c:
        if e[channels_index] == 9:
          new_chord.add(tuple(e))

    new_chord = [list(e) for e in new_chord]

    new_chord.sort(key = lambda x: (-x[pitches_index], x[patches_index]))

    fixed_chordified_score.append(new_chord)

  return fixed_chordified_score, bad_chords_counter, duplicate_pitches_counter

#===============================================================================

def score_chord_to_tones_chord(chord,
                               transpose_value=0,
                               channels_index=3,
                               pitches_index=4):

  return sorted(set([(p[4]+transpose_value) % 12 for p in chord if p[channels_index] != 9]))

#===============================================================================

def enhanced_chord_to_tones_chord(enhanced_chord):
  return sorted(set([t[4] % 12 for t in enhanced_chord if t[3] != 9]))

#===============================================================================

def transpose_tones_chord(tones_chord, transpose_value=0):
  return sorted([((60+t)+transpose_value) % 12 for t in sorted(set(tones_chord))])

#===============================================================================

def transpose_pitches_chord(pitches_chord, transpose_value=0):
  return [max(1, min(127, p+transpose_value)) for p in sorted(set(pitches_chord), reverse=True)]

#===============================================================================

def normalize_chord_durations(chord, 
                              dur_idx=2, 
                              norm_factor=100
                              ):

  nchord = copy.deepcopy(chord)
  
  for c in nchord:
    c[dur_idx] = int(round(max(1 / norm_factor, c[dur_idx] // norm_factor) * norm_factor))

  return nchord

#===============================================================================

def normalize_chordified_score_durations(chordified_score, 
                                         dur_idx=2, 
                                         norm_factor=100
                                         ):

  ncscore = copy.deepcopy(chordified_score)
  
  for cc in ncscore:
    for c in cc:
      c[dur_idx] = int(round(max(1 / norm_factor, c[dur_idx] // norm_factor) * norm_factor))

  return ncscore

#===============================================================================

def tones_chord_type(tones_chord, 
                     return_chord_type_index=True,
                     use_filtered_chords=False,
                     use_full_chords=True
                     ):

  WN = WHITE_NOTES
  BN = BLACK_NOTES
  MX = WHITE_NOTES + BLACK_NOTES

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  tones_chord = sorted(tones_chord)

  ctype = 'UNKNOWN'

  if tones_chord in CHORDS:

    if sorted(set(tones_chord) & set(WN)) == tones_chord:
      ctype = 'WHITE'

    elif sorted(set(tones_chord) & set(BN)) == tones_chord:
      ctype = 'BLACK'

    if len(tones_chord) > 1 and sorted(set(tones_chord) & set(MX)) == tones_chord:

      if len(sorted(set(tones_chord) & set(WN))) == len(sorted(set(tones_chord) & set(BN))):
        ctype = 'MIXED GRAY'

      elif len(sorted(set(tones_chord) & set(WN))) > len(sorted(set(tones_chord) & set(BN))):
        ctype = 'MIXED WHITE'

      elif len(sorted(set(tones_chord) & set(WN))) < len(sorted(set(tones_chord) & set(BN))):
        ctype = 'MIXED BLACK'

  if return_chord_type_index:
    return CHORDS_TYPES.index(ctype)

  else:
    return ctype

#===============================================================================

def find_best_tones_chord(src_tones_chords,
                          trg_tones_chords,
                          find_longest=True
                          ):

  not_seen_trg_chords = []

  max_len = 0

  for tc in trg_tones_chords:
    if sorted(tc) in src_tones_chords:
      not_seen_trg_chords.append(sorted(tc))
      max_len = max(max_len, len(tc))

  if not not_seen_trg_chords:
    max_len = len(max(trg_tones_chords, key=len))
    not_seen_trg_chords = trg_tones_chords

  if find_longest:
    return random.choice([c for c in not_seen_trg_chords if len(c) == max_len])

  else:
    return random.choice(not_seen_trg_chords)

#===============================================================================

def find_matching_tones_chords(tones_chord,
                               matching_chord_length=-1,
                               match_chord_type=True,
                               use_filtered_chords=True,
                               use_full_chords=True
                               ):

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  tones_chord = sorted(tones_chord)

  tclen = len(tones_chord)

  tctype = tones_chord_type(tones_chord, use_filtered_chords=use_filtered_chords)

  matches = []

  for tc in CHORDS:

    if matching_chord_length == -1:
      if len(tc) > tclen:
        if sorted(lists_intersections(tc, tones_chord)) == tones_chord:
          if match_chord_type:
            if tones_chord_type(tc, use_filtered_chords=use_filtered_chords) == tctype:
              tcdiffs = lists_sym_differences(tc, tones_chord)
              if all(tone_type(d) == tctype % 3 for d in tcdiffs):
                matches.append(tc)
          else:
            matches.append(tc)

    else:

      if len(tc) == max(tclen, matching_chord_length):
        if sorted(lists_intersections(tc, tones_chord)) == tones_chord:
          if match_chord_type:
            if tones_chord_type(tc, use_filtered_chords=use_filtered_chords) == tctype:
              tcdiffs = lists_sym_differences(tc, tones_chord)
              if all(tone_type(d) == tctype % 3 for d in tcdiffs):
                matches.append(tc)
          else:
            matches.append(tc)

  return sorted(matches, key=len)

#===============================================================================

def check_and_fix_pitches_chord(pitches_chord,
                                remove_duplicate_pitches=True,
                                use_filtered_chords=False,
                                use_full_chords=True,
                                fix_bad_pitches=False,
                                ):
  
  if remove_duplicate_pitches:
    pitches_chord = sorted(set(pitches_chord), reverse=True)
  else:
    pitches_chord = sorted(pitches_chord, reverse=True)

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  chord = copy.deepcopy(pitches_chord)
    
  tones_chord = sorted(set([t % 12 for t in chord]))

  if tones_chord:

      if tones_chord not in CHORDS:
        
        if len(tones_chord) == 2:
          tones_counts = Counter([p % 12 for p in pitches_chord]).most_common()

          if tones_counts[0][1] > 1:
            tones_chord = [tones_counts[0][0]]
          
          elif tones_counts[1][1] > 1:
            tones_chord = [tones_counts[1][0]]
          
          else:
            tones_chord = [pitches_chord[0] % 12]

        else:
          tones_chord_combs = [list(comb) for i in range(len(tones_chord)-1, 0, -1) for comb in combinations(tones_chord, i)]

          for co in tones_chord_combs:
            if co in CHORDS:
              tones_chord = co
              break

          if len(tones_chord) == 1:
            tones_chord = [pitches_chord[0] % 12]
              
  chord.sort(reverse=True)

  new_chord = set()
  pipa = []

  for e in chord:
    if e % 12 in tones_chord:
      new_chord.add(tuple([e]))
      pipa.append(e)

    elif (e+1) % 12 in tones_chord:
      e += 1
      new_chord.add(tuple([e]))
      pipa.append(e)

    elif (e-1) % 12 in tones_chord:
      e -= 1
      new_chord.add(tuple([e]))
      pipa.append(e)

  if fix_bad_pitches:

    bad_chord = set()

    for e in chord:
    
      if e % 12 not in tones_chord:
        bad_chord.add(tuple([e]))
      
      elif (e+1) % 12 not in tones_chord:
        bad_chord.add(tuple([e]))
      
      elif (e-1) % 12 not in tones_chord:
        bad_chord.add(tuple([e]))
          
    for bc in bad_chord:

      bc = list(bc)

      tone = find_closest_tone(tones_chord, bc[0] % 12)

      new_pitch = ((bc[0] // 12) * 12) + tone

      if new_pitch not in pipa:
        new_chord.add(tuple([new_pitch]))
        pipa.append(new_pitch)

  new_pitches_chord = [e[0] for e in new_chord]

  return sorted(new_pitches_chord, reverse=True)

#===============================================================================

def tones_chords_to_types(tones_chords, 
                          return_chord_type_index=False
                          ):

  types_tones_chords = []

  for c in tones_chords:

    c.sort()

    ctype = tones_chord_type(c, return_chord_type_index=return_chord_type_index)

    types_tones_chords.append(ctype)

  return types_tones_chords

#===============================================================================

def morph_tones_chord(tones_chord, 
                      trg_tone, 
                      use_filtered_chords=True,
                      use_full_chords=True
                      ):

  src_tones_chord = sorted(sorted(set(tones_chord)) + [trg_tone])

  combs = [list(comb) for i in range(len(src_tones_chord), 0, -1) for comb in combinations(src_tones_chord, i) if trg_tone in list(comb)]

  matches = []

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  for c in combs:
    if sorted(set(c)) in CHORDS:
      matches.append(sorted(set(c)))

  max_len = len(max(matches, key=len))

  return random.choice([m for m in matches if len(m) == max_len])

#===============================================================================

def chord_to_pchord(chord, channels_index=3, pitches_index=4):

  pchord = []

  for cc in chord:
    if cc[channels_index] != 9:
      pchord.append(cc[pitches_index])

  return pchord

#===============================================================================

def chordified_score_pitches(chordified_score, 
                             mode='dominant',
                             return_tones=False,
                             omit_drums=True,
                             score_patch=-1,
                             channels_index=3,
                             pitches_index=4,
                             patches_index=6                          
                            ):

  results = []

  for c in chordified_score:
    
    if -1 < score_patch < 128:
      ptcs = sorted([e[pitches_index] for e in c if e[channels_index] != 9 and e[patches_index] == score_patch], reverse=True)
    
    else:
      ptcs = sorted([e[pitches_index] for e in c if e[channels_index] != 9], reverse=True)

    if ptcs:

      if mode == 'dominant':
        
        mtone = statistics.mode([p % 12 for p in ptcs])
        
        if return_tones:
          results.append(mtone)
        
        else:
          results.append(sorted(set([p for p in ptcs if p % 12 == mtone]), reverse=True))
      
      elif mode == 'high':
        
        if return_tones:
          results.append(ptcs[0] % 12)

        else:
          results.append([ptcs[0]])

      elif mode == 'base':

        if return_tones:
          results.append(ptcs[-1] % 12)

        else:
          results.append([ptcs[-1]])

      elif mode == 'average':

        if return_tones:
          results.append(statistics.mean(ptcs) % 12)

        else:
          results.append([statistics.mean(ptcs)])

      else:

        mtone = statistics.mode([p % 12 for p in ptcs])
        
        if return_tones:
          results.append(mtone)
        
        else:
          results.append(sorted(set([p for p in ptcs if p % 12 == mtone]), reverse=True))

    else:

      if not omit_drums:
        
        if return_tones:
          results.append(-1)
        
        else:
          results.append([-1])

  return results

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of chords module
#=======================================================================================================