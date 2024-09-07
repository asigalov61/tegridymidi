#=======================================================================================================
# Tegridy MIDI Plots module
#=======================================================================================================

from tegridymidi.helpers import generate_colors, add_arrays

import matplotlib.pyplot as plt

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
