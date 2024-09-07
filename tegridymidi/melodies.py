#=======================================================================================================
# Melodies module
#=======================================================================================================

import copy
from collections import Counter

from tegridymidi.processors import chordify_score
from tegridymidi.misc import adjust_score_velocities
from tegridymidi.constants import ALL_CHORDS_FULL
from tegridymidi.helpers import stack_list

#===============================================================================

def fix_monophonic_score_durations(monophonic_score):
  
    fixed_score = []

    if monophonic_score[0][0] == 'note':

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][1]

        if note[1]+note[2] >= nmt:
          note_dur = nmt-note[1]-1
        else:
          note_dur = note[2]

        new_note = [note[0], note[1], note_dur] + note[3:]

        fixed_score.append(new_note)

      fixed_score.append(monophonic_score[-1])

    elif type(monophonic_score[0][0]) == int:

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][0]

        if note[0]+note[1] >= nmt:
          note_dur = nmt-note[0]-1
        else:
          note_dur = note[1]

        new_note = [note[0], note_dur] + note[2:]

        fixed_score.append(new_note)

      fixed_score.append(monophonic_score[-1]) 

    return fixed_score

#===============================================================================

def extract_melody(chordified_enhanced_score, 
                    melody_range=[48, 84], 
                    melody_channel=0,
                    melody_patch=0,
                    melody_velocity=0,
                    stacked_melody=False,
                    stacked_melody_base_pitch=60
                  ):

    if stacked_melody:

      
      all_pitches_chords = []
      for e in chordified_enhanced_score:
        all_pitches_chords.append(sorted(set([p[4] for p in e]), reverse=True))
      
      melody_score = []
      for i, chord in enumerate(chordified_enhanced_score):

        if melody_velocity > 0:
          vel = melody_velocity
        else:
          vel = chord[0][5]

        melody_score.append(['note', chord[0][1], chord[0][2], melody_channel, stacked_melody_base_pitch+(stack_list([p % 12 for p in all_pitches_chords[i]]) % 12), vel, melody_patch])
  
    else:

      melody_score = copy.deepcopy([c[0] for c in chordified_enhanced_score if c[0][3] != 9])
      
      for e in melody_score:
        
          e[3] = melody_channel

          if melody_velocity > 0:
            e[5] = melody_velocity

          e[6] = melody_patch

          if e[4] < melody_range[0]:
              e[4] = (e[4] % 12) + melody_range[0]
              
          if e[4] >= melody_range[1]:
              e[4] = (e[4] % 12) + (melody_range[1]-12)

    return fix_monophonic_score_durations(melody_score)

#===============================================================================

def create_enhanced_monophonic_melody(monophonic_melody):

    enhanced_monophonic_melody = []

    for i, note in enumerate(monophonic_melody[:-1]):

      enhanced_monophonic_melody.append(note)

      if note[1]+note[2] < monophonic_melody[i+1][1]:
        
        delta_time = monophonic_melody[i+1][1] - (note[1]+note[2])
        enhanced_monophonic_melody.append(['silence', note[1]+note[2], delta_time, note[3], 0, 0, note[6]])
        
    enhanced_monophonic_melody.append(monophonic_melody[-1])

    return enhanced_monophonic_melody

#===============================================================================

def frame_monophonic_melody(monophonic_melody, min_frame_time_threshold=10):

    mzip = list(zip(monophonic_melody[:-1], monophonic_melody[1:]))

    times_counts = Counter([(b[1]-a[1]) for a, b in mzip]).most_common()

    mc_time = next((item for item, count in times_counts if item >= min_frame_time_threshold), min_frame_time_threshold)

    times = [(b[1]-a[1]) // mc_time for a, b in mzip] + [monophonic_melody[-1][2] // mc_time]

    framed_melody = []

    for i, note in enumerate(monophonic_melody):
      
      stime = note[1]
      count = times[i]
      
      if count != 0:
        for j in range(count):

          new_note = copy.deepcopy(note)
          new_note[1] = stime + (j * mc_time)
          new_note[2] = mc_time
          framed_melody.append(new_note)
      
      else:
        framed_melody.append(note)

    return [framed_melody, mc_time]

#===============================================================================

def add_melody_to_enhanced_score_notes(enhanced_score_notes,
                                      melody_start_time=0,
                                      melody_start_chord=0,
                                      melody_notes_min_duration=-1,
                                      melody_notes_max_duration=255,
                                      melody_duration_overlap_tolerance=4,
                                      melody_avg_duration_divider=2,
                                      melody_base_octave=5,
                                      melody_channel=3,
                                      melody_patch=40,
                                      melody_max_velocity=110,
                                      acc_max_velocity=90,
                                      pass_drums=True,
                                      return_melody=False
                                      ):
  
    if pass_drums:
      score = copy.deepcopy(enhanced_score_notes)
    else:
      score = [e for e in copy.deepcopy(enhanced_score_notes) if e[3] !=9]

    if melody_notes_min_duration > 0:
      min_duration = melody_notes_min_duration
    else:
      durs = [d[2] for d in score]
      min_duration = Counter(durs).most_common()[0][0]

    adjust_score_velocities(score, acc_max_velocity)

    cscore = chordify_score([1000, score])

    melody_score = []
    acc_score = []

    pt = melody_start_time

    for c in cscore[:melody_start_chord]:
      acc_score.extend(c)

    for c in cscore[melody_start_chord:]:

      durs = [d[2] if d[3] != 9 else -1 for d in c]

      if not all(d == -1 for d in durs):
        ndurs = [d for d in durs if d != -1]
        avg_dur = (sum(ndurs) / len(ndurs)) / melody_avg_duration_divider
        best_dur = min(durs, key=lambda x:abs(x-avg_dur))
        pidx = durs.index(best_dur)

        cc = copy.deepcopy(c[pidx])

        if c[0][1] >= pt - melody_duration_overlap_tolerance and best_dur >= min_duration:

          cc[3] = melody_channel
          cc[4] = (c[pidx][4] % 24)
          cc[5] = 100 + ((c[pidx][4] % 12) * 2)
          cc[6] = melody_patch

          melody_score.append(cc)
          acc_score.extend(c)

          pt = c[0][1]+c[pidx][2]

        else:
          acc_score.extend(c)

      else:
        acc_score.extend(c)

    values = [e[4] % 24 for e in melody_score]
    smoothed = [values[0]]
    for i in range(1, len(values)):
        if abs(smoothed[-1] - values[i]) >= 12:
            if smoothed[-1] < values[i]:
                smoothed.append(values[i] - 12)
            else:
                smoothed.append(values[i] + 12)
        else:
            smoothed.append(values[i])

    smoothed_melody = copy.deepcopy(melody_score)

    for i, e in enumerate(smoothed_melody):
      e[4] = (melody_base_octave * 12) + smoothed[i]

    for i, m in enumerate(smoothed_melody[1:]):
      if m[1] - smoothed_melody[i][1] < melody_notes_max_duration:
        smoothed_melody[i][2] = m[1] - smoothed_melody[i][1]

    adjust_score_velocities(smoothed_melody, melody_max_velocity)

    if return_melody:
      final_score = sorted(smoothed_melody, key=lambda x: (x[1], -x[4]))

    else:
      final_score = sorted(smoothed_melody + acc_score, key=lambda x: (x[1], -x[4]))

    return final_score

#===============================================================================

def harmonize_enhanced_melody_score_notes(enhanced_melody_score_notes):
  
  mel_tones = [e[4] % 12 for e in enhanced_melody_score_notes]

  cur_chord = []

  song = []

  for i, m in enumerate(mel_tones):
    cur_chord.append(m)
    cc = sorted(set(cur_chord))

    if cc in ALL_CHORDS_FULL:
      song.append(cc)

    else:
      while sorted(set(cur_chord)) not in ALL_CHORDS_FULL:
        cur_chord.pop(0)
      cc = sorted(set(cur_chord))
      song.append(cc)

  return song

#===============================================================================

def split_melody(enhanced_melody_score_notes, 
                 split_time=-1, 
                 max_score_time=255
                 ):

  mel_chunks = []

  if split_time == -1:

    durs = [max(0, min(max_score_time, e[2])) for e in enhanced_melody_score_notes]
    stime = max(durs)
    
  else:
    stime = split_time

  pe = enhanced_melody_score_notes[0]
  chu = []
  
  for e in enhanced_melody_score_notes:
    dtime = max(0, min(max_score_time, e[1]-pe[1]))

    if dtime > max(durs):
      if chu:
        mel_chunks.append(chu)
      chu = []
      chu.append(e)
    else:
      chu.append(e)

    pe = e

  if chu:
    mel_chunks.append(chu)

  return mel_chunks, [[m[0][1], m[-1][1]] for m in mel_chunks], len(mel_chunks)

#===============================================================================

def harmonize_enhanced_melody_score_notes_to_ms_SONG(escore_notes,
                                                      melody_velocity=-1,
                                                      melody_channel=3,
                                                      melody_patch=40,
                                                      melody_base_octave=4,
                                                      harmonized_tones_chords_velocity=-1,
                                                      harmonized_tones_chords_channel=0,
                                                      harmonized_tones_chords_patch=0
                                                    ):

  harmonized_tones_chords = harmonize_enhanced_melody_score_notes(escore_notes)

  harm_escore_notes = []

  time = 0

  for i, note in enumerate(escore_notes):

    time = note[1]
    dur = note[2]
    ptc = note[4]

    if melody_velocity == -1:
      vel = int(110 + ((ptc % 12) * 1.5))
    else:
      vel = melody_velocity

    harm_escore_notes.append(['note', time, dur, melody_channel, ptc, vel, melody_patch])

    for t in harmonized_tones_chords[i]:

      ptc = (melody_base_octave * 12) + t

      if harmonized_tones_chords_velocity == -1:
        vel = int(80 + ((ptc % 12) * 1.5))
      else:
        vel = harmonized_tones_chords_velocity

      harm_escore_notes.append(['note', time, dur, harmonized_tones_chords_channel, ptc, vel, harmonized_tones_chords_patch])

  return sorted(harm_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of melodies module
#=======================================================================================================
