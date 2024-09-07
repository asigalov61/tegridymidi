# Module tegridymidi

??? example "View Source"
        #=======================================================================================================

        # Tegridy MIDI __init__ module

        #=======================================================================================================

        #=======================================================================================================

        # CPU Imports

        #=======================================================================================================

        from .bits_and_ints import *

        from .chords import *

        from .constants import *

        from .files_helpers import *

        from .haystack_search import *

        from .helpers import *

        from .karaoke import *

        from .matrixes import *

        from .melodies import *

        from .midi_to_colab_audio import *

        from .misc import *

        from .plots import *

        from .processors import *

        from .sample_midis_helpers import *

        from .tokenizers import *

        #=======================================================================================================

        # GPU Imports

        #=======================================================================================================

        try:

          import torch

          if torch.cuda.is_available():

            from .x_transformer import *

            import random

        except ImportError:

          pass

        #=======================================================================================================

        # This is the end of __init__ module

        #=======================================================================================================

## Sub-modules

* [tegridymidi.bits_and_ints](bits_and_ints/)
* [tegridymidi.chords](chords/)
* [tegridymidi.constants](constants/)
* [tegridymidi.files_helpers](files_helpers/)
* [tegridymidi.haystack_search](haystack_search/)
* [tegridymidi.helpers](helpers/)
* [tegridymidi.karaoke](karaoke/)
* [tegridymidi.legacy](legacy/)
* [tegridymidi.matrixes](matrixes/)
* [tegridymidi.melodies](melodies/)
* [tegridymidi.midi_to_colab_audio](midi_to_colab_audio/)
* [tegridymidi.misc](misc/)
* [tegridymidi.plots](plots/)
* [tegridymidi.processors](processors/)
* [tegridymidi.sample_midis_helpers](sample_midis_helpers/)
* [tegridymidi.tokenizers](tokenizers/)
* [tegridymidi.x_transformer](x_transformer/)

## Variables

```python3
ALL_CHORDS
```

```python3
ALL_CHORDS_FILTERED
```

```python3
ALL_CHORDS_FULL
```

```python3
ALL_CHORDS_GROUPED
```

```python3
ALL_CHORDS_PAIRS_FILTERED
```

```python3
ALL_CHORDS_PAIRS_SORTED
```

```python3
ALL_CHORDS_SORTED
```

```python3
ALL_CHORDS_TRANS
```

```python3
ALL_CHORDS_TRIPLETS_FILTERED
```

```python3
ALL_CHORDS_TRIPLETS_SORTED
```

```python3
ALL_MELODIES
```

```python3
ALL_PITCHES_CHORDS_FILTERED
```

```python3
ALL_PITCHES_CHORDS_SORTED
```

```python3
All_events
```

```python3
BLACK_NOTES
```

```python3
CHORDS_TYPES
```

```python3
DEFAULT_MODE
```

```python3
Event2channelindex
```

```python3
FLUIDSETTING_EXISTS
```

```python3
FLUID_FAILED
```

```python3
FLUID_OK
```

```python3
FLUID_PLAYER_DONE
```

```python3
FLUID_PLAYER_PLAYING
```

```python3
FLUID_PLAYER_READY
```

```python3
FLUID_PLAYER_STOPPING
```

```python3
FLUID_PLAYER_TEMPO_EXTERNAL_BPM
```

```python3
FLUID_PLAYER_TEMPO_EXTERNAL_MIDI
```

```python3
FLUID_PLAYER_TEMPO_INTERNAL
```

```python3
MIDI_Instruments_Families
```

```python3
MIDI_events
```

```python3
Meta_events
```

```python3
NULL_SYMBOL
```

```python3
Nontext_meta_events
```

```python3
Notenum2percussion
```

```python3
Number2patch
```

```python3
PP_INDENT
```

```python3
RTLD_GLOBAL
```

```python3
RTLD_LOCAL
```

```python3
Text_events
```

```python3
Version
```

```python3
VersionDate
```

```python3
WHITE_NOTES
```

```python3
api_version
```

```python3
fluid_synth_get_channel_info
```

```python3
fluid_synth_get_chorus_depth_ms
```

```python3
fluid_synth_get_chorus_speed_Hz
```

```python3
fluid_synth_set_chorus_full
```

```python3
fluid_synth_set_midi_router
```

```python3
fluid_synth_set_reverb_full
```

```python3
lib
```