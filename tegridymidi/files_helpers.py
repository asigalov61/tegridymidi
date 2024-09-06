#=======================================================================================================
# Files helpers functions
#=======================================================================================================

import os

#=======================================================================================================

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

#=======================================================================================================

def print_directory_tree(root_dir, exclude_hidden=True, output_file=None):
    lines = build_directory_tree(root_dir, exclude_hidden=exclude_hidden)
    if output_file:
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    else:
        for line in lines:
            print(line)

#=======================================================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of files_helpers module
#=======================================================================================================
