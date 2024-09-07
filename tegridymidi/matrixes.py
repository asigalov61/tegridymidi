#=======================================================================================================
# Matrixes module
#=======================================================================================================

from collections import Counter

#===============================================================================

def create_similarity_matrix(list_of_values, matrix_length=0):

    counts = Counter(list_of_values).items()

    if matrix_length > 0:
      sim_matrix = [0] * max(matrix_length, len(list_of_values))
    else:
      sim_matrix = [0] * len(counts)

    for c in counts:
      sim_matrix[c[0]] = c[1]

    similarity_matrix = [[0] * len(sim_matrix) for _ in range(len(sim_matrix))]

    for i in range(len(sim_matrix)):
      for j in range(len(sim_matrix)):
        if max(sim_matrix[i], sim_matrix[j]) != 0:
          similarity_matrix[i][j] = min(sim_matrix[i], sim_matrix[j]) / max(sim_matrix[i], sim_matrix[j])

    return similarity_matrix, sim_matrix

#===============================================================================

def escore_notes_to_escore_matrix(escore_notes,
                                  alt_velocities=False,
                                  flip_matrix=False,
                                  reverse_matrix=False
                                  ):

  last_time = escore_notes[-1][1]
  last_notes = [e for e in escore_notes if e[1] == last_time]
  max_last_dur = max([e[2] for e in last_notes])

  time_range = last_time+max_last_dur

  channels_list = sorted(set([e[3] for e in escore_notes]))

  escore_matrixes = []

  for cha in channels_list:

    escore_matrix = [[[-1, -1]] * 128 for _ in range(time_range)]

    pe = escore_notes[0]

    for i, note in enumerate(escore_notes):

        etype, time, duration, channel, pitch, velocity, patch = note

        time = max(0, time)
        duration = max(1, duration)
        channel = max(0, min(15, channel))
        pitch = max(0, min(127, pitch))
        velocity = max(0, min(127, velocity))
        patch = max(0, min(128, patch))

        if alt_velocities:
            velocity -= (i % 2)

        if channel == cha:

          for t in range(time, min(time + duration, time_range)):

            escore_matrix[t][pitch] = [velocity, patch]

        pe = note

    if flip_matrix:

      temp_matrix = []

      for m in escore_matrix:
        temp_matrix.append(m[::-1])

      escore_matrix = temp_matrix

    if reverse_matrix:
      escore_matrix = escore_matrix[::-1]

    escore_matrixes.append(escore_matrix)

  return [channels_list, escore_matrixes]

#===============================================================================

def escore_matrix_to_merged_escore_notes(full_escore_matrix,
                                        max_note_duration=4000
                                        ):

  merged_escore_notes = []

  mat_channels_list = full_escore_matrix[0]
  
  for m, cha in enumerate(mat_channels_list):

    escore_matrix = full_escore_matrix[1][m]

    result = []

    for j in range(len(escore_matrix[0])):

        count = 1

        for i in range(1, len(escore_matrix)):

          if escore_matrix[i][j] != [-1, -1] and escore_matrix[i][j][1] == escore_matrix[i-1][j][1] and count < max_note_duration:
              count += 1

          else:
              if count > 1:  
                result.append([i-count, count, j, escore_matrix[i-1][j]])

              count = 1

        if count > 1:
            result.append([len(escore_matrix)-count, count, j, escore_matrix[-1][j]])

    result.sort(key=lambda x: (x[0], -x[2]))

    for r in result:
      merged_escore_notes.append(['note', r[0], r[1], cha, r[2], r[3][0], r[3][1]])

  return sorted(merged_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

#===============================================================================

def escore_matrix_to_original_escore_notes(full_escore_matrix):

  merged_escore_notes = []

  mat_channels_list = full_escore_matrix[0]

  for m, cha in enumerate(mat_channels_list):

    escore_matrix = full_escore_matrix[1][m]

    result = []

    for j in range(len(escore_matrix[0])):

        count = 1

        for i in range(1, len(escore_matrix)):

          if escore_matrix[i][j] != [-1, -1] and escore_matrix[i][j] == escore_matrix[i-1][j]:
              count += 1

          else:
              if count > 1:
                result.append([i-count, count, j, escore_matrix[i-1][j]])

              count = 1

        if count > 1:
            result.append([len(escore_matrix)-count, count, j, escore_matrix[-1][j]])

    result.sort(key=lambda x: (x[0], -x[2]))

    for r in result:
      merged_escore_notes.append(['note', r[0], r[1], cha, r[2], r[3][0], r[3][1]])

  return sorted(merged_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

#===============================================================================

def escore_notes_to_binary_matrix(escore_notes, 
                                  channel=0, 
                                  patch=0,
                                  flip_matrix=False,
                                  reverse_matrix=False
                                  ):

  escore = [e for e in escore_notes if e[3] == channel and e[6] == patch]

  if escore:
    last_time = escore[-1][1]
    last_notes = [e for e in escore if e[1] == last_time]
    max_last_dur = max([e[2] for e in last_notes])

    time_range = last_time+max_last_dur

    escore_matrix = []

    escore_matrix = [[0] * 128 for _ in range(time_range)]

    for note in escore:

        etype, time, duration, chan, pitch, velocity, pat = note

        time = max(0, time)
        duration = max(1, duration)
        chan = max(0, min(15, chan))
        pitch = max(0, min(127, pitch))
        velocity = max(0, min(127, velocity))
        pat = max(0, min(128, pat))

        if channel == chan and patch == pat:

          for t in range(time, min(time + duration, time_range)):

            escore_matrix[t][pitch] = 1

    if flip_matrix:

      temp_matrix = []

      for m in escore_matrix:
        temp_matrix.append(m[::-1])

      escore_matrix = temp_matrix

    if reverse_matrix:
      escore_matrix = escore_matrix[::-1]

    return escore_matrix

  else:
    return None

#===============================================================================

def binary_matrix_to_original_escore_notes(binary_matrix, 
                                           channel=0, 
                                           patch=0, 
                                           velocity=-1
                                           ):

  result = []

  for j in range(len(binary_matrix[0])):

      count = 1

      for i in range(1, len(binary_matrix)):

        if binary_matrix[i][j] != 0 and binary_matrix[i][j] == binary_matrix[i-1][j]:
            count += 1

        else:
          if count > 1:
            result.append([i-count, count, j, binary_matrix[i-1][j]])
          
          else:
            if binary_matrix[i-1][j] != 0:
              result.append([i-count, count, j, binary_matrix[i-1][j]])

          count = 1

      if count > 1:
          result.append([len(binary_matrix)-count, count, j, binary_matrix[-1][j]])
      
      else:
        if binary_matrix[i-1][j] != 0:
          result.append([i-count, count, j, binary_matrix[i-1][j]])

  result.sort(key=lambda x: (x[0], -x[2]))

  original_escore_notes = []

  vel = velocity

  for r in result:
    
    if velocity == -1:
      vel = max(40, r[2])

    original_escore_notes.append(['note', r[0], r[1], channel, r[2], vel, patch])

  return sorted(original_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of matrixes module
#=======================================================================================================
