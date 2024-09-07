#=======================================================================================================
#
# Tegridy MIDI Python Module
#
# Version 24.9.6
#
# Project Los Angeles
# Tegridy Code 2024
#
#=======================================================================================================

from tegridymidi.sample_midis_helpers import *

import statistics
import copy

#=======================================================================================================

def Optimus_Squash(chords_list, simulate_velocity=True, mono_compression=False):

  '''Input: Flat chords list
            Simulate velocity or not
            Mono-compression enabled or disabled
            
            Default is almost lossless 25% compression, otherwise, lossy 50% compression (mono-compression)

     Output: Squashed chords list
             Resulting compression level

             Please note that if drums are passed through as is

     Project Los Angeles
     Tegridy Code 2021'''

  output = []
  ptime = 0
  vel = 0
  boost = 15
  stptc = []
  ocount = 0
  rcount = 0

  for c in chords_list:
    
    cc = copy.deepcopy(c)
    ocount += 1
    
    if [cc[1], cc[3], (cc[4] % 12) + 60] not in stptc:
      stptc.append([cc[1], cc[3], (cc[4] % 12) + 60])

      if cc[3] != 9:
        cc[4] = (c[4] % 12) + 60

      if simulate_velocity and c[1] != ptime:
        vel = c[4] + boost
      
      if cc[3] != 9:
        cc[5] = vel

      if mono_compression:
        if c[1] != ptime:
          output.append(cc)
          rcount += 1  
      else:
        output.append(cc)
        rcount += 1
      
      ptime = c[1]

  output.sort(key=lambda x: (x[1], x[4]))

  comp_level = 100 - int((rcount * 100) / ocount)

  return output, comp_level

###################################################################################

def Optimus_Signature(chords_list, calculate_full_signature=False):

    '''Optimus Signature

    ---In the name of the search for a perfect score slice signature---
     
    Input: Flat chords list to evaluate

    Output: Full Optimus Signature as a list
            Best/recommended Optimus Signature as a list

    Project Los Angeles
    Tegridy Code 2021'''
    
    # Pitches

    ## StDev
    if calculate_full_signature:
      psd = statistics.stdev([int(y[4]) for y in chords_list])
    else:
      psd = 0

    ## Median
    pmh = statistics.median_high([int(y[4]) for y in chords_list])
    pm = statistics.median([int(y[4]) for y in chords_list])
    pml = statistics.median_low([int(y[4]) for y in chords_list])
    
    ## Mean
    if calculate_full_signature:
      phm = statistics.harmonic_mean([int(y[4]) for y in chords_list])
    else:
      phm = 0

    # Durations
    dur = statistics.median([int(y[2]) for y in chords_list])

    # Velocities

    vel = statistics.median([int(y[5]) for y in chords_list])

    # Beats
    mtds = statistics.median([int(abs(chords_list[i-1][1]-chords_list[i][1])) for i in range(1, len(chords_list))])
    if calculate_full_signature:
      hmtds = statistics.harmonic_mean([int(abs(chords_list[i-1][1]-chords_list[i][1])) for i in range(1, len(chords_list))])
    else:
      hmtds = 0

    # Final Optimus signatures
    full_Optimus_signature = [round(psd), round(pmh), round(pm), round(pml), round(phm), round(dur), round(vel), round(mtds), round(hmtds)]
    ########################    PStDev     PMedianH    PMedian    PMedianL    PHarmoMe    Duration    Velocity      Beat       HarmoBeat

    best_Optimus_signature = [round(pmh), round(pm), round(pml), round(dur, -1), round(vel, -1), round(mtds, -1)]
    ########################   PMedianH    PMedian    PMedianL      Duration        Velocity          Beat
    
    # Return...
    return full_Optimus_signature, best_Optimus_signature

#=======================================================================================================

__all__ = [name for name in globals() if not name.startswith('_')]

#=======================================================================================================
# This is the end of tegridymidi module
#=======================================================================================================
