#=======================================================================================================
# Tegridy MIDI __init__ module
#=======================================================================================================

#=======================================================================================================
# CPU Imports
#=======================================================================================================

from .misc import *
from .helpers import *
from .processors import *
from .melodies import *
from .chords import *
from .constants import *
from .matrixes import *
from .plots import *
from .haystack_search import *
from .midi_to_colab_audio import *
from .files_helpers import *
from .sample_midis_helpers import *

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
