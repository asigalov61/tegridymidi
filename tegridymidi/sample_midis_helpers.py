# Sample MIDIs helpers

import pkg_resources
import os

def get_sample_midis_path():
    """
    Returns the path to the sample_midis directory.
    """
    return pkg_resources.resource_filename('tegridymidi', 'sample_midis')

def list_sample_midis():
    """
    Lists all files in the sample_midis directory.
    """
    sample_midis_path = get_sample_midis_path()
    return os.listdir(sample_midis_path)

def list_sample_midis_with_full_paths():
    """
    Lists all files in the sample_midis directory with their full paths.
    """
    sample_midis_path = get_sample_midis_path()
    return [os.path.join(sample_midis_path, file) for file in os.listdir(sample_midis_path)]
