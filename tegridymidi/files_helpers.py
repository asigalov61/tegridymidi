#=======================================================================================================
# Files helpers functions
#=======================================================================================================

import os
from datetime import datetime
import pickle
import hashlib

#===============================================================================

def build_directory_tree(root_dir, indent='', exclude_hidden=True, is_root=True):
    lines = []
    if is_root:
        lines.append(os.path.basename(root_dir) + '/')
        indent += '    '
    items = [item for item in os.listdir(root_dir) if not (exclude_hidden and item.startswith('.'))]
    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)
        if index == len(items) - 1:
            line = indent + '└── ' + item
            new_indent = indent + '    '
        else:
            line = indent + '├── ' + item
            new_indent = indent + '│   '
        lines.append(line)
        if os.path.isdir(path):
            lines.extend(build_directory_tree(path, new_indent, exclude_hidden, is_root=False))
    return lines

#===============================================================================

def print_directory_tree(root_dir, exclude_hidden=True, output_file=None):
    lines = build_directory_tree(root_dir, exclude_hidden=exclude_hidden)
    if output_file:
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    else:
        for line in lines:
            print(line)

#===============================================================================

def file_time_stamp(input_file_name='File_Created_on_', ext = ''):

  '''Tegridy File Time Stamp
     
  Input: Full path and file name without extention
         File extension
          
  Output: File name string with time-stamp and extension (time-stamped file name)

  Project Los Angeles
  Tegridy Code 2021'''       

  print('Time-stamping output file...')

  now = ''
  now_n = str(datetime.now())
  now_n = now_n.replace(' ', '_')
  now_n = now_n.replace(':', '_')
  now = now_n.replace('.', '_')
      
  fname = input_file_name + str(now) + ext

  return(fname)

#===============================================================================

def pickle_file_writer(Data, input_file_name='TMIDI_Pickle_File'):

  '''Tegridy Pickle File Writer
     
  Input: Data to write (I.e. a list)
         Full path and file name without extention
         
  Output: Named Pickle file

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy Pickle File Writer')

  full_path_to_output_dataset_to = input_file_name + '.pickle'

  if os.path.exists(full_path_to_output_dataset_to):
    os.remove(full_path_to_output_dataset_to)
    print('Removing old Dataset...')
  else:
    print("Creating new Dataset file...")

  with open(full_path_to_output_dataset_to, 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(Data, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

  print('Dataset was saved as:', full_path_to_output_dataset_to)
  print('Task complete. Enjoy! :)')

#===============================================================================

def pitckle_file_reader(input_file_name='TMIDI_Pickle_File', ext='.pickle', verbose=True):

  '''Tegridy Pickle File Loader
     
  Input: Full path and file name with or without extention
         File extension if different from default .pickle
       
  Output: Standard Python 3 unpickled data object

  Project Los Angeles
  Tegridy Code 2021'''

  if verbose:
    print('Tegridy Pickle File Loader')
    print('Loading the pickle file. Please wait...')

  if os.path.basename(input_file_name).endswith(ext):
    fname = input_file_name
  
  else:
    fname = input_file_name + ext

  with open(fname, 'rb') as pickle_file:
    content = pickle.load(pickle_file)

  if verbose:
    print('Done!')

  return content

#===============================================================================

def md5_hash(file_path_or_data=None, original_md5_hash=None):

  if type(file_path_or_data) == str:

    with open(file_path_or_data, 'rb') as file_to_check:
      data = file_to_check.read()
      
      if data:
        md5 = hashlib.md5(data).hexdigest()

  else:
    if file_path_or_data:
      md5 = hashlib.md5(file_path_or_data).hexdigest()

  if md5:

    if original_md5_hash:

      if md5 == original_md5_hash:
        check = True
      else:
        check = False
        
    else:
      check = None

    return [md5, check]

  else:

    md5 = None
    check = None

    return [md5, check]

#===============================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of files_helpers module
#=======================================================================================================
