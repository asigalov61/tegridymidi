#=======================================================================================================
# Matrixes module
#=======================================================================================================

from collections import Counter

from tegridymidi.helpers import find_value_power

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

def compress_binary_matrix(binary_matrix, 
                           only_compress_zeros=False,
                           return_compression_ratio=False
                           ):

  compressed_bmatrix = []

  zm = [0] * len(binary_matrix[0])
  pm = [0] * len(binary_matrix[0])

  mcount = 0

  for m in binary_matrix:
    
    if only_compress_zeros:
      if m != zm:
        compressed_bmatrix.append(m)
        mcount += 1
    
    else:
      if m != pm:
        compressed_bmatrix.append(m)
        mcount += 1
    
    pm = m

  if return_compression_ratio:
    return [compressed_bmatrix, mcount / len(binary_matrix)]

  else:
    return compressed_bmatrix

#===============================================================================

def cubic_kernel(x):
    abs_x = abs(x)
    if abs_x <= 1:
        return 1.5 * abs_x**3 - 2.5 * abs_x**2 + 1
    elif abs_x <= 2:
        return -0.5 * abs_x**3 + 2.5 * abs_x**2 - 4 * abs_x + 2
    else:
        return 0

#===============================================================================

def resize_matrix(matrix, new_height, new_width):
    old_height = len(matrix)
    old_width = len(matrix[0])
    resized_matrix = [[0] * new_width for _ in range(new_height)]
    
    for i in range(new_height):
        for j in range(new_width):
            old_i = i * old_height / new_height
            old_j = j * old_width / new_width
            
            value = 0
            total_weight = 0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    i_m = min(max(int(old_i) + m, 0), old_height - 1)
                    j_n = min(max(int(old_j) + n, 0), old_width - 1)
                    
                    if matrix[i_m][j_n] == 0:
                        continue
                    
                    weight = cubic_kernel(old_i - i_m) * cubic_kernel(old_j - j_n)
                    value += matrix[i_m][j_n] * weight
                    total_weight += weight
            
            if total_weight > 0:
                value /= total_weight
            
            resized_matrix[i][j] = int(value > 0.5)
    
    return resized_matrix

#===============================================================================

def square_binary_matrix(binary_matrix, 
                         matrix_size=128,
                         use_fast_squaring=False,
                         return_plot_points=False
                         ):

  if use_fast_squaring:

    step = round(len(binary_matrix) / matrix_size)

    samples = []

    for i in range(0, len(binary_matrix), step):
      samples.append(tuple([tuple(d) for d in binary_matrix[i:i+step]]))

    resized_matrix = []

    zmatrix = [[0] * matrix_size]

    for s in samples:

      samples_counts = Counter(s).most_common()

      best_sample = tuple([0] * matrix_size)
      pm = tuple(zmatrix[0])

      for sc in samples_counts:
        if sc[0] != tuple(zmatrix[0]) and sc[0] != pm:
          best_sample = sc[0]
          pm = sc[0]
          break
        
        pm = sc[0]

      resized_matrix.append(list(best_sample))

    resized_matrix = resized_matrix[:matrix_size]
    resized_matrix += zmatrix * (matrix_size - len(resized_matrix))
    
  else:
    resized_matrix = resize_matrix(binary_matrix, matrix_size, matrix_size)

  points = [(i, j) for i in range(matrix_size) for j in range(matrix_size) if resized_matrix[i][j] == 1]

  if return_plot_points:
    return [resized_matrix, points]

  else:
    return resized_matrix

#===============================================================================

def mean(matrix):
    return sum(sum(row) for row in matrix) / (len(matrix) * len(matrix[0]))

#===============================================================================

def variance(matrix, mean_value):
    return sum(sum((element - mean_value) ** 2 for element in row) for row in matrix) / (len(matrix) * len(matrix[0]))
    
#===============================================================================

def covariance(matrix1, matrix2, mean1, mean2):
    return sum(sum((matrix1[i][j] - mean1) * (matrix2[i][j] - mean2) for j in range(len(matrix1[0]))) for i in range(len(matrix1))) / (len(matrix1) * len(matrix1[0]))

#===============================================================================

def ssim_index(matrix1, matrix2, bit_depth=1):

    if len(matrix1) != len(matrix2) and len(matrix1[0]) != len(matrix2[0]):
      return -1

    K1, K2 = 0.01, 0.03
    L = bit_depth
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    mu1 = mean(matrix1)
    mu2 = mean(matrix2)
    
    sigma1_sq = variance(matrix1, mu1)
    sigma2_sq = variance(matrix2, mu2)
    
    sigma12 = covariance(matrix1, matrix2, mu1, mu2)
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim

#===============================================================================

def find_most_similar_matrix(array_of_matrices, 
                             trg_matrix,
                             matrices_bit_depth=1,
                             return_most_similar_index=False
                             ):
   
    max_ssim = -float('inf')
    most_similar_index = -1

    for i, matrix in enumerate(array_of_matrices):

        ssim = ssim_index(matrix, trg_matrix, bit_depth=matrices_bit_depth)
        
        if ssim > max_ssim:
            max_ssim = ssim
            most_similar_index = i
    
    if return_most_similar_index:
      return most_similar_index
    
    else:
      return array_of_matrices[most_similar_index]

#===============================================================================

def escore_notes_to_image_matrix(escore_notes,
                                  num_img_channels=3,
                                  filter_out_zero_rows=False,
                                  filter_out_duplicate_rows=False,
                                  flip_matrix=False,
                                  reverse_matrix=False
                                  ):

  escore_notes = sorted(escore_notes, key=lambda x: (x[1], x[6]))

  if num_img_channels > 1:
    n_mat_channels = 3
  else:
    n_mat_channels = 1

  if escore_notes:
    last_time = escore_notes[-1][1]
    last_notes = [e for e in escore_notes if e[1] == last_time]
    max_last_dur = max([e[2] for e in last_notes])

    time_range = last_time+max_last_dur

    escore_matrix = []

    escore_matrix = [[0] * 128 for _ in range(time_range)]

    for note in escore_notes:

        etype, time, duration, chan, pitch, velocity, pat = note

        time = max(0, time)
        duration = max(2, duration)
        chan = max(0, min(15, chan))
        pitch = max(0, min(127, pitch))
        velocity = max(0, min(127, velocity))
        patch = max(0, min(128, pat))

        if chan != 9:
          pat = patch + 128
        else:
          pat = 127

        seen_pats = []

        for t in range(time, min(time + duration, time_range)):

          mat_value = escore_matrix[t][pitch]

          mat_value_0 = (mat_value // (256 * 256)) % 256
          mat_value_1 = (mat_value // 256) % 256

          cur_num_chans = 0

          if 0 < mat_value < 256 and pat not in seen_pats:
            cur_num_chans = 1
          elif 256 < mat_value < (256 * 256) and pat not in seen_pats:
            cur_num_chans = 2

          if cur_num_chans < n_mat_channels:

            if n_mat_channels == 1:

              escore_matrix[t][pitch] = pat
              seen_pats.append(pat)

            elif n_mat_channels == 3:

              if cur_num_chans == 0:
                escore_matrix[t][pitch] = pat
                seen_pats.append(pat)
              elif cur_num_chans == 1:
                escore_matrix[t][pitch] = (256 * 256 * mat_value_0) + (256 * pat)
                seen_pats.append(pat)
              elif cur_num_chans == 2:
                escore_matrix[t][pitch] = (256 * 256 * mat_value_0) + (256 * mat_value_1) + pat
                seen_pats.append(pat)

    if filter_out_zero_rows:
      escore_matrix = [e for e in escore_matrix if sum(e) != 0]

    if filter_out_duplicate_rows:

      dd_escore_matrix = []

      pr = [-1] * 128
      for e in escore_matrix:
        if e != pr:
          dd_escore_matrix.append(e)
          pr = e
      
      escore_matrix = dd_escore_matrix

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

def image_matrix_to_original_escore_notes(image_matrix,
                                          velocity=-1
                                          ):

  result = []

  for j in range(len(image_matrix[0])):

      count = 1

      for i in range(1, len(image_matrix)):

        if image_matrix[i][j] != 0 and image_matrix[i][j] == image_matrix[i-1][j]:
            count += 1

        else:
          if count > 1:
            result.append([i-count, count, j, image_matrix[i-1][j]])

          else:
            if image_matrix[i-1][j] != 0:
              result.append([i-count, count, j, image_matrix[i-1][j]])

          count = 1

      if count > 1:
          result.append([len(image_matrix)-count, count, j, image_matrix[-1][j]])

      else:
        if image_matrix[i-1][j] != 0:
          result.append([i-count, count, j, image_matrix[i-1][j]])

  result.sort(key=lambda x: (x[0], -x[2]))

  original_escore_notes = []

  vel = velocity

  for r in result:

    if velocity == -1:
      vel = max(40, r[2])

    ptc0 = 0
    ptc1 = 0
    ptc2 = 0

    if find_value_power(r[3], 256) == 0:
      ptc0 = r[3] % 256

    elif find_value_power(r[3], 256) == 1:
      ptc0 = r[3] // 256
      ptc1 = (r[3] // 256) % 256

    elif find_value_power(r[3], 256) == 2:
      ptc0 = (r[3] // 256) // 256
      ptc1 = (r[3] // 256) % 256
      ptc2 = r[3] % 256

    ptcs = [ptc0, ptc1, ptc2]
    patches = [p for p in ptcs if p != 0]

    for i, p in enumerate(patches):

      if p < 128:
        patch = 128
        channel = 9

      else:
        patch = p % 128
        chan = p // 8

        if chan == 9:
          chan += 1

        channel = min(15, chan)

      original_escore_notes.append(['note', r[0], r[1], channel, r[2], vel, patch])

  output_score = sorted(original_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

  adjust_score_velocities(output_score, 127)

  return output_score

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of matrixes module
#=======================================================================================================
