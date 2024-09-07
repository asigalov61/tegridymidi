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

***

### Project Los Angeles
### Tegridy Code 2024
