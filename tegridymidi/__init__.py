#=======================================================================================================
# Tegridy MIDI __init__ module
#=======================================================================================================

#=======================================================================================================
# CPU Imports
#=======================================================================================================

from bits_and_ints import *
from chords import *
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
