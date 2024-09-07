#=======================================================================================================
# Tegridy MIDI Plots module
#=======================================================================================================

from tegridymidi.helpers import generate_colors, add_arrays, normalize_to_range

import os
from collections import Counter
from itertools import groupby

import numpy as np

import networkx as nx

from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.decomposition import PCA

from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import zscore

import matplotlib.pyplot as plt

#===============================================================================

def find_closest_points(points, return_points=True):

  """
  Find closest 2D points
  """

  coords = np.array(points)

  num_points = coords.shape[0]
  closest_matches = np.zeros(num_points, dtype=int)
  distances = np.zeros((num_points, num_points))

  for i in range(num_points):
      for j in range(num_points):
          if i != j:
              distances[i, j] = np.linalg.norm(coords[i] - coords[j])
          else:
              distances[i, j] = np.inf

  closest_matches = np.argmin(distances, axis=1)
  
  if return_points:
    points_matches = coords[closest_matches].tolist()
    return points_matches
  
  else:
    return closest_matches.tolist()

#===============================================================================

def reduce_dimensionality_tsne(list_of_valies,
                                n_comp=2,
                                n_iter=5000,
                                verbose=True
                              ):

  """
  Reduces the dimensionality of the values using t-SNE.
  """

  vals = np.array(list_of_valies)

  tsne = TSNE(n_components=n_comp,
              n_iter=n_iter,
              verbose=verbose)

  reduced_vals = tsne.fit_transform(vals)

  return reduced_vals.tolist()

#===============================================================================

def compute_mst_edges(similarity_scores_list):

  """
  Computes the Minimum Spanning Tree (MST) edges based on the similarity scores.
  """
  
  num_tokens = len(similarity_scores_list[0])

  graph = nx.Graph()

  for i in range(num_tokens):
      for j in range(i + 1, num_tokens):
          weight = 1 - similarity_scores_list[i][j]
          graph.add_edge(i, j, weight=weight)

  mst = nx.minimum_spanning_tree(graph)

  mst_edges = list(mst.edges(data=False))

  return mst_edges

#===============================================================================

def square_matrix_points_colors(square_matrix_points):

  """
  Returns colors for square matrix points
  """

  cmap = generate_colors(12)

  chords = []
  chords_dict = set()
  counts = []

  for k, v in groupby(square_matrix_points, key=lambda x: x[0]):
    pgroup = [vv[1] for vv in v]
    chord = sorted(set(pgroup))
    tchord = sorted(set([p % 12 for p in chord]))
    chords_dict.add(tuple(tchord))
    chords.append(tuple(tchord))
    counts.append(len(pgroup))

  chords_dict = sorted(chords_dict)

  colors = []

  for i, c in enumerate(chords):
    colors.extend([cmap[round(sum(c) / len(c))]] * counts[i])

  return colors

#===============================================================================

def remove_points_outliers(points, z_score_threshold=3):

  points = np.array(points)

  z_scores = np.abs(zscore(points, axis=0))

  return points[(z_scores < z_score_threshold).all(axis=1)].tolist()

#===============================================================================

def generate_labels(lists_of_values, 
                    return_indices_labels=False
                    ):

  ordered_indices = list(range(len(lists_of_values)))
  ordered_indices_labels = [str(i) for i in ordered_indices]
  ordered_values_labels = [str(lists_of_values[i]) for i in ordered_indices]

  if return_indices_labels:
    return ordered_indices_labels
  
  else:
    return ordered_values_labels

#===============================================================================

def reduce_dimensionality_pca(list_of_values, n_components=2):

  """
  Reduces the dimensionality of the values using PCA.
  """

  pca = PCA(n_components=n_components)
  pca_data = pca.fit_transform(list_of_values)
  
  return pca_data.tolist()

def reduce_dimensionality_simple(list_of_values, 
                                 return_means=True,
                                 return_std_devs=True,
                                 return_medians=False,
                                 return_vars=False
                                 ):
  
  '''
  Reduces dimensionality of the values in a simple way
  '''

  array = np.array(list_of_values)
  results = []

  if return_means:
      means = np.mean(array, axis=1)
      results.append(means)

  if return_std_devs:
      std_devs = np.std(array, axis=1)
      results.append(std_devs)

  if return_medians:
      medians = np.median(array, axis=1)
      results.append(medians)

  if return_vars:
      vars = np.var(array, axis=1)
      results.append(vars)

  merged_results = np.column_stack(results)
  
  return merged_results.tolist()

#===============================================================================

def reduce_dimensionality_2d_distance(list_of_values, p=5):

  '''
  Reduces the dimensionality of the values using 2d distance
  '''

  values = np.array(list_of_values)

  dist_matrix = distance_matrix(values, values, p=p)

  mst = minimum_spanning_tree(dist_matrix).toarray()

  points = []

  for i in range(len(values)):
      for j in range(len(values)):
          if mst[i, j] > 0:
              points.append([i, j])

  return points

#===============================================================================

def reduce_dimensionality_simple_pca(list_of_values, n_components=2):

  '''
  Reduces the dimensionality of the values using simple PCA
  '''

  reduced_values = []

  for l in list_of_values:

    norm_values = [round(v * len(l)) for v in normalize_to_range(l, (n_components+1) // 2)]

    pca_values = Counter(norm_values).most_common()
    pca_values = [vv[0] / len(l) for vv in pca_values]
    pca_values = pca_values[:n_components]
    pca_values = pca_values + [0] * (n_components - len(pca_values))

    reduced_values.append(pca_values)

  return reduced_values

#===============================================================================

def find_shortest_constellation_path(points, 
                                     start_point_idx, 
                                     end_point_idx,
                                     p=5,
                                     return_path_length=False,
                                     return_path_points=False,
                                     ):

    """
    Finds the shortest path between two points of the points constellation
    """

    points = np.array(points)

    dist_matrix = distance_matrix(points, points, p=p)

    mst = minimum_spanning_tree(dist_matrix).toarray()

    G = nx.Graph()

    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    path = nx.shortest_path(G, 
                            source=start_point_idx, 
                            target=end_point_idx, 
                            weight='weight'
                            )
    
    path_length = nx.shortest_path_length(G, 
                                          source=start_point_idx, 
                                          target=end_point_idx, 
                                          weight='weight')
        
    path_points = points[np.array(path)].tolist()


    if return_path_points:
      return path_points

    if return_path_length:
      return path_length

    return path

#===============================================================================

def plot_square_matrix_points(list_of_points,
                              list_of_points_colors,
                              plot_size=(7, 7),
                              point_size = 10,
                              show_grid_lines=False,
                              plot_title = 'Square Matrix Points Plot',
                              return_plt=False,
                              save_plt='',
                              save_only_plt_image=True,
                              save_transparent=False
                              ):

  '''Square matrix points plot'''

  fig, ax = plt.subplots(figsize=plot_size)

  ax.set_facecolor('black')

  if show_grid_lines:
    ax.grid(color='white')

  plt.xlabel('Time Step', c='black')
  plt.ylabel('MIDI Pitch', c='black')

  plt.title(plot_title)

  plt.scatter([p[0] for p in list_of_points], 
              [p[1] for p in list_of_points], 
              c=list_of_points_colors, 
              s=point_size
              )

  if save_plt != '':
    if save_only_plt_image:
      plt.axis('off')
      plt.title('')
      plt.savefig(save_plt, 
                  transparent=save_transparent, 
                  bbox_inches='tight', 
                  pad_inches=0, 
                  facecolor='black'
                  )
      plt.close()
    
    else:
      plt.savefig(save_plt)
      plt.close()

  if return_plt:
    return fig

  plt.show()
  plt.close()

#===============================================================================

def plot_cosine_similarities(lists_of_values,
                             plot_size=(7, 7),
                             save_plot=''
                            ):

  """
  Cosine similarities plot
  """

  cos_sim = metrics.pairwise_distances(lists_of_values, metric='cosine')

  plt.figure(figsize=plot_size)

  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")

  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]

  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)

  plt.xlabel("Index")
  plt.ylabel("Index")

  plt.tight_layout()

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()
  plt.close()

#===============================================================================

def plot_points_with_mst_lines(points, 
                               points_labels, 
                               points_mst_edges,
                               plot_size=(20, 20),
                               labels_size=24,
                               save_plot=''
                               ):

  """
  Plots 2D points with labels and MST lines.
  """

  plt.figure(figsize=plot_size)

  for i, label in enumerate(points_labels):
      plt.scatter(points[i][0], points[i][1])
      plt.annotate(label, (points[i][0], points[i][1]), fontsize=labels_size)

  for edge in points_mst_edges:
      i, j = edge
      plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'k-', alpha=0.5)

  plt.title('Points Map with MST Lines', fontsize=labels_size)
  plt.xlabel('X-axis', fontsize=labels_size)
  plt.ylabel('Y-axis', fontsize=labels_size)

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()

  plt.close()

#===============================================================================

def plot_points_constellation(points, 
                              points_labels,
                              p=5,                              
                              plot_size=(15, 15),
                              labels_size=12,
                              show_grid=False,
                              save_plot=''
                              ):

  """
  Plots 2D points constellation
  """

  points = np.array(points)

  dist_matrix = distance_matrix(points, points, p=p)

  mst = minimum_spanning_tree(dist_matrix).toarray()

  plt.figure(figsize=plot_size)

  plt.scatter(points[:, 0], points[:, 1], color='blue')

  for i, label in enumerate(points_labels):
      plt.annotate(label, (points[i, 0], points[i, 1]), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=labels_size
                   )

  for i in range(len(points)):
      for j in range(len(points)):
          if mst[i, j] > 0:
              plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k--')

  plt.xlabel('X-axis', fontsize=labels_size)
  plt.ylabel('Y-axis', fontsize=labels_size)
  plt.title('2D Coordinates with Minimum Spanning Tree', fontsize=labels_size)

  plt.grid(show_grid)

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()

  plt.close()

#===============================================================================

def plot_parsons_code(parsons_code, 
                      start_pitch=60, 
                      return_plot_dict=False, 
                      return_plot_string=False,
                      plot_size=(10, 10),
                      labels_size=16,
                      save_plot=''
                      ):
  
  '''
  Plot parsons code string
  '''

  if parsons_code[0] != "*":
      return None

  contour_dict = {}
  pitch = 0
  index = 0

  maxp = 0
  minp = 0

  contour_dict[(pitch, index)] = "*"

  for point in parsons_code:
      if point == "R":
          index += 1
          contour_dict[(pitch, index)] = "-"

          index += 1
          contour_dict[(pitch, index)] = "*"
          
      elif point == "U":
          index += 1
          pitch -= 1
          contour_dict[(pitch, index)] = "/"

          index += 1
          pitch -= 1
          contour_dict[(pitch, index)] = "*"

          if pitch < maxp:
              maxp = pitch

      elif point == "D":
          index += 1
          pitch += 1
          contour_dict[(pitch, index)] = "\\"

          index += 1
          pitch += 1
          contour_dict[(pitch, index)] = "*"

          if pitch > minp:
              minp = pitch

  if return_plot_dict:
    return contour_dict
  
  if return_plot_string:

    plot_string = ''

    for pitch in range(maxp, minp+1):
        line = [" " for _ in range(index + 1)]
        for pos in range(index + 1):
            if (pitch, pos) in contour_dict:
                line[pos] = contour_dict[(pitch, pos)]

        plot_string = "".join(line)

    return plot_string

  labels = []
  pitches = []
  positions = []
  cur_pitch = start_pitch
  pitch_idx = 0

  for k, v in contour_dict.items():

    if v != '*':

      pitches.append(cur_pitch)
      positions.append(pitch_idx)

      if v == '/':
        cur_pitch += 1
        labels.append('U')
      
      elif v == '\\':
        cur_pitch -= 1
        labels.append('D')

      elif v == '-':
        labels.append('R')

      pitch_idx += 1

  plt.figure(figsize=plot_size)

  
  plt.plot(pitches)

  for i, point in enumerate(zip(positions, pitches)):
    plt.annotate(labels[i], point, fontsize=labels_size)
  

  plt.title('Parsons Code with Labels', fontsize=labels_size)
  plt.xlabel('Position', fontsize=labels_size)
  plt.ylabel('Pitch', fontsize=labels_size)

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()

  plt.close()

#===============================================================================

def plot_ms_SONG(ms_song,
                  preview_length_in_notes=0,
                  block_lines_times_list = None,
                  plot_title='ms Song',
                  max_num_colors=129, 
                  drums_color_num=128, 
                  plot_size=(11,4), 
                  note_height = 0.75,
                  show_grid_lines=False,
                  return_plt = False,
                  timings_multiplier=1,
                  save_plt='',
                  save_only_plt_image=True,
                  save_transparent=False
                  ):

  '''Tegridy ms SONG plotter/vizualizer'''

  notes = [s for s in ms_song if s[0] == 'note']

  if (len(max(notes, key=len)) != 7) and (len(min(notes, key=len)) != 7):
    print('The song notes do not have patches information')
    print('Ploease add patches to the notes in the song')

  else:

    start_times = [(s[1] * timings_multiplier) / 1000 for s in notes]
    durations = [(s[2]  * timings_multiplier) / 1000 for s in notes]
    pitches = [s[4] for s in notes]
    patches = [s[6] for s in notes]

    colors = generate_colors(max_num_colors)
    colors[drums_color_num] = (1, 1, 1)

    pbl = (notes[preview_length_in_notes][1] * timings_multiplier) / 1000

    fig, ax = plt.subplots(figsize=plot_size)
    #fig, ax = plt.subplots()

    # Create a rectangle for each note with color based on patch number
    for start, duration, pitch, patch in zip(start_times, durations, pitches, patches):
        rect = plt.Rectangle((start, pitch), duration, note_height, facecolor=colors[patch])
        ax.add_patch(rect)

    # Set the limits of the plot
    ax.set_xlim([min(start_times), max(add_arrays(start_times, durations))])
    ax.set_ylim([min(pitches)-1, max(pitches)+1])

    # Set the background color to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')

    if preview_length_in_notes > 0:
      ax.axvline(x=pbl, c='white')

    if block_lines_times_list:
      for bl in block_lines_times_list:
        ax.axvline(x=bl, c='white')
           
    if show_grid_lines:
      ax.grid(color='white')

    plt.xlabel('Time (s)', c='black')
    plt.ylabel('MIDI Pitch', c='black')

    plt.title(plot_title)

    if save_plt != '':
      if save_only_plt_image:
        plt.axis('off')
        plt.title('')
        plt.savefig(save_plt, transparent=save_transparent, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
      
      else:
        plt.savefig(save_plt)
        plt.close()

    if return_plt:
      return fig

    plt.show()
    plt.close()

#=======================================================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of plots module
#=======================================================================================================
