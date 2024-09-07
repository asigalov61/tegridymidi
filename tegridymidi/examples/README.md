# Tegridy MIDI Examples

***

## MIDI Input

```sh
from tegridymidi import processors, chords, melodies
from tegridymidi import sample_midis_helpers

#===============================================================================
# Input MIDI file (as filepath or bytes)

input_midi = [path for path in sample_midis_helpers.list_sample_midis_with_full_paths() if 'seed2.mid' in path][0]

#===============================================================================
# Raw single-track ms score

raw_score = processors.midi2single_track_ms_score(input_midi)

#===============================================================================
# Enhanced score notes

escore_notes = processors.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]

#===============================================================================
# Augmented enhanced score notes

escore_notes = processors.augment_enhanced_score_notes(escore_notes)

#===============================================================================
# Chordified augmented enhanced score

cscore = chords.chordify_score([1000, escore_notes])

#===============================================================================
# Monophonic melody score with adjusted durations and custom patch (40 / Violin)

melody = melodies.fix_monophonic_score_durations(melodies.extract_melody(cscore, melody_patch=40))
```

## MIDI Output

```sh
import copy
from IPython.display import display, Audio

from tegridymidi import processors, misc, plots
from tegridymidi import midi_to_colab_audio

#===============================================================================
# Input variables

output_file_name = './melody' # output MIDI file path without extension
output_score = copy.deepcopy(melody) # or escore_notes (any notes list in TMIDIX/MIDI score format)

print('=' * 70)

#===============================================================================
# Creating output score patches list from the enhanced score notes

# This is optional and you can skip this if you do not care about patches
# Alternatively You can use TMIDIX functions to detect all composition patches

# This function will create a single track patch list (16 patches)
# patches = TMIDIX.patch_list_from_enhanced_score_notes(output_score)

# And this functions will create a full patch list (any number of patches)
# It will also patch the score if there are more than 16 patches
# This function is preferred and recommended

output_score, patches, overflow_patches = misc.patch_enhanced_score_notes(output_score)

#===============================================================================
# Converting to MIDI

detailed_stats = processors.ms_escore_notes2midi(output_score,
                                                          output_signature = 'TMIDIX MIDI Composition',
                                                          output_file_name = output_file_name,
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=patches,
                                                          timings_multiplier=16 # Restoring augmented timings
                                                          )

#===============================================================================
# Printing resulting MIDI stats

print('=' * 70)
print(detailed_stats)

#===============================================================================
# Rendering MIDI to (raw) audio for listening to and for further processing

print('=' * 70)
print('Converting MIDI to audio...Please wait...')
midi_audio = midi_to_colab_audio.midi_to_colab_audio('./melody.mid')
display(Audio(midi_audio, rate=16000, normalize=False))

#===============================================================================
# Resulting MIDI plot

plots.plot_ms_escore_notes(output_score, plot_title=output_file_name+'.mid')
```

***

### Project Los Angeles
### Tegridy Code 2024
