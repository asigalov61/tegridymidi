#=======================================================================================================
# Tegridy MIDI Tokenizers module
#=======================================================================================================

import copy
from collections import Counter
from itertools import combinations

from tegridymidi.constants import ALL_CHORDS_FILTERED, ALL_CHORDS_SORTED, ALL_CHORDS_FULL
from tegridymidi.helpers import flatten

#===============================================================================


def basic_enhanced_delta_score_notes_tokenizer(enhanced_delta_score_notes,
                                              tokenize_start_times=True,
                                              tokenize_durations=True,
                                              tokenize_channels=True,
                                              tokenize_pitches=True,
                                              tokenize_velocities=True,
                                              tokenize_patches=True,
                                              score_timings_range=256,
                                              max_seq_len=-1,
                                              seq_pad_value=-1
                                              ):
  
  
  
  score_tokens_ints_seq = []

  tokens_shifts = [-1] * 7

  for d in enhanced_delta_score_notes:

    seq = []
    shift = 0

    if tokenize_start_times:
      seq.append(d[0])
      tokens_shifts[0] = shift
      shift += score_timings_range

    if tokenize_durations:
      seq.append(d[1]+shift)
      tokens_shifts[1] = shift
      shift += score_timings_range

    if tokenize_channels:
      tokens_shifts[2] = shift
      seq.append(d[2]+shift)
      shift += 16
    
    if tokenize_pitches:
      tokens_shifts[3] = shift
      seq.append(d[3]+shift)
      shift += 128
    
    if tokenize_velocities:
      tokens_shifts[4] = shift
      seq.append(d[4]+shift)
      shift += 128

    if tokenize_patches:
      tokens_shifts[5] = shift
      seq.append(d[5]+shift)
      shift += 129

    tokens_shifts[6] = shift
    score_tokens_ints_seq.append(seq)

  final_score_tokens_ints_seq = flatten(score_tokens_ints_seq)

  if max_seq_len > -1:
    final_score_tokens_ints_seq = final_score_tokens_ints_seq[:max_seq_len]

  if seq_pad_value > -1:
    final_score_tokens_ints_seq += [seq_pad_value] * (max_seq_len - len(final_score_tokens_ints_seq))

  return [score_tokens_ints_seq,
          final_score_tokens_ints_seq, 
          tokens_shifts,
          seq_pad_value, 
          max_seq_len,
          len(score_tokens_ints_seq),
          len(final_score_tokens_ints_seq)
          ]

###################################################################################

def basic_enhanced_delta_score_notes_detokenizer(tokenized_seq, 
                                                 tokens_shifts, 
                                                 timings_multiplier=16
                                                 ):

  song_f = []

  time = 0
  dur = 16
  channel = 0
  pitch = 60
  vel = 90
  pat = 0

  note_seq_len = len([t for t in tokens_shifts if t > -1])-1
  tok_shifts_idxs = [i for i in range(len(tokens_shifts[:-1])) if tokens_shifts[i] > - 1]

  song = []

  for i in range(0, len(tokenized_seq), note_seq_len):
    note = tokenized_seq[i:i+note_seq_len]
    song.append(note)

  for note in song:
    for i, idx in enumerate(tok_shifts_idxs):
      if idx == 0:
        time += (note[i]-tokens_shifts[0]) * timings_multiplier
      elif idx == 1:
        dur = (note[i]-tokens_shifts[1]) * timings_multiplier
      elif idx == 2:
        channel = (note[i]-tokens_shifts[2])
      elif idx == 3:
        pitch = (note[i]-tokens_shifts[3])
      elif idx == 4:
        vel = (note[i]-tokens_shifts[4])
      elif idx == 5:
        pat = (note[i]-tokens_shifts[5])

    song_f.append(['note', time, dur, channel, pitch, vel, pat ])

  return song_f

###################################################################################

def enhanced_chord_to_chord_token(enhanced_chord, 
                                  channels_index=3, 
                                  pitches_index=4, 
                                  use_filtered_chords=False,
                                  use_full_chords=True
                                  ):
  
  bad_chords_counter = 0
  duplicate_pitches_counter = 0

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  tones_chord = sorted(set([t[pitches_index] % 12 for t in enhanced_chord if t[channels_index] != 9]))

  original_tones_chord = copy.deepcopy(tones_chord)

  if tones_chord:

      if tones_chord not in CHORDS:
        
        pitches_chord = sorted(set([p[pitches_index] for p in enhanced_chord if p[channels_index] != 9]), reverse=True)
        
        if len(tones_chord) == 2:
          tones_counts = Counter([p % 12 for p in pitches_chord]).most_common()

          if tones_counts[0][1] > 1:
            tones_chord = [tones_counts[0][0]]
          elif tones_counts[1][1] > 1:
            tones_chord = [tones_counts[1][0]]
          else:
            tones_chord = [pitches_chord[0] % 12]

        else:
          tones_chord_combs = [list(comb) for i in range(len(tones_chord)-2, 0, -1) for comb in combinations(tones_chord, i+1)]

          for co in tones_chord_combs:
            if co in CHORDS:
              tones_chord = co
              break

  if use_filtered_chords:
    chord_token = ALL_CHORDS_FILTERED.index(tones_chord)
  else:
    chord_token = ALL_CHORDS_SORTED.index(tones_chord)

  return [chord_token, tones_chord, original_tones_chord, sorted(set(original_tones_chord) ^ set(tones_chord))]

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of chords module
#=======================================================================================================