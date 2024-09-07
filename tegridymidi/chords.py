#=======================================================================================================
# Tegridy MIDI Chords module
#=======================================================================================================

def chordify_score(score,
                  return_choridfied_score=True,
                  return_detected_score_information=False
                  ):

    if score:
    
      num_tracks = 1
      single_track_score = []
      score_num_ticks = 0

      if type(score[0]) == int and len(score) > 1:

        score_type = 'MIDI_PY'
        score_num_ticks = score[0]

        while num_tracks < len(score):
            for event in score[num_tracks]:
              single_track_score.append(event)
            num_tracks += 1
      
      else:
        score_type = 'CUSTOM'
        single_track_score = score

      if single_track_score and single_track_score[0]:
        
        try:

          if type(single_track_score[0][0]) == str or single_track_score[0][0] == 'note':
            single_track_score.sort(key = lambda x: x[1])
            score_timings = [s[1] for s in single_track_score]
          else:
            score_timings = [s[0] for s in single_track_score]

          is_score_time_absolute = lambda sct: all(x <= y for x, y in zip(sct, sct[1:]))

          score_timings_type = ''

          if is_score_time_absolute(score_timings):
            score_timings_type = 'ABS'

            chords = []
            cho = []

            if score_type == 'MIDI_PY':
              pe = single_track_score[0]
            else:
              pe = single_track_score[0]

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
                ptime = pe[1]
              else:
                time = e[0]
                ptime = pe[0]

              if time == ptime:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

              pe = e

            if len(cho) > 0:
              chords.append(cho)

          else:
            score_timings_type = 'REL'
            
            chords = []
            cho = []

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
              else:
                time = e[0]

              if time == 0:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

            if len(cho) > 0:
              chords.append(cho)

          requested_data = []

          if return_detected_score_information:
            
            detected_score_information = []

            detected_score_information.append(['Score type', score_type])
            detected_score_information.append(['Score timings type', score_timings_type])
            detected_score_information.append(['Score tpq', score_num_ticks])
            detected_score_information.append(['Score number of tracks', num_tracks])
            
            requested_data.append(detected_score_information)

          if return_choridfied_score and return_detected_score_information:
            requested_data.append(chords)

          if return_choridfied_score and not return_detected_score_information:
            requested_data.extend(chords)

          return requested_data

        except Exception as e:
          print('Error!')
          print('Check score for consistency and compatibility!')
          print('Exception detected:', e)

      else:
        return None

    else:
      return None

#=======================================================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of chords module
#=======================================================================================================
