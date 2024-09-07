#=======================================================================================================
# Tegridy MIDI Bits and Ints module
#=======================================================================================================

from tegridymidi.chords import bad_chord

#=======================================================================================================

def tones_chord_to_bits(chord, reverse=True):

    bits = [0] * 12

    for num in chord:
        bits[num] = 1
    
    if reverse:
      bits.reverse()
      return bits
    
    else:
      return bits

#===============================================================================

def bits_to_tones_chord(bits):
    return [i for i, bit in enumerate(bits) if bit == 1]

#===============================================================================

def shift_bits(bits, n):
    return bits[-n:] + bits[:-n]

#===============================================================================

def bits_to_int(bits, shift_bits_value=0):
    bits = shift_bits(bits, shift_bits_value)
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    
    return result

#===============================================================================

def int_to_bits(n):
    bits = [0] * 12
    for i in range(12):
        bits[11 - i] = n % 2
        n //= 2
    
    return bits

#===============================================================================

def pitches_chord_to_int(pitches_chord, tones_transpose_value=0):

    pitches_chord = [x for x in pitches_chord if 0 < x < 128]

    if not (-12 < tones_transpose_value < 12):
      tones_transpose_value = 0

    tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))
    bits = tones_chord_to_bits(tones_chord)
    integer = bits_to_int(bits, shift_bits_value=tones_transpose_value)

    return integer

#===============================================================================

def int_to_pitches_chord(integer, chord_base_pitch=60): 
    if 0 < integer < 4096:
      bits = int_to_bits(integer)
      tones_chord = bits_to_tones_chord(bits)
      if not bad_chord(tones_chord):
        pitches_chord = [t+chord_base_pitch for t in tones_chord]
        return [pitches_chord, tones_chord]
      
      else:
        return 0 # Bad chord code
    
    else:
      return -1 # Bad integer code

#===============================================================================

def tones_chords_to_bits(tones_chords):

  bits_tones_chords = []

  for c in tones_chords:

    c.sort()

    bits = tones_chord_to_bits(c)

    bits_tones_chords.append(bits)

  return bits_tones_chords

#===============================================================================

def tones_chords_to_ints(tones_chords):

  ints_tones_chords = []

  for c in tones_chords:

    c.sort()

    bits = tones_chord_to_bits(c)

    number = bits_to_int(bits)

    ints_tones_chords.append(number)

  return ints_tones_chords

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of bits_and_ints module
#=======================================================================================================
