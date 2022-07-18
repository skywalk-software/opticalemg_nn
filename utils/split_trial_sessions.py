import argparse
from boltons.fileutils import iter_find_files
import os
import numpy as np
import h5py

def is_h5(f):
    return f.endswith(".h5")

def get_files(dir_path):
    if os.path.isdir(dir_path):
        files = iter_find_files(dir_path, "*.h5", include_dirs=True)
    else:
        files = [input]
    files = [os.path.abspath(f) for f in files]
    files = [f for f in files if is_h5(f)]
    return files

def split_session(f, remove_original=False):
    with h5py.File(f, "r") as h5:
        sessions_list = list(h5.keys())
        for v in ["metadata", "__DATA_TYPES__"]:
            if v in sessions_list:
                sessions_list.remove(v)
        if not sessions_list[0].isdigit():
            print("Skipping {}: this file appears to already contain only a single session".format(f))
            return
        for i, sess in enumerate(sessions_list):
            new_file = f.replace(".h5", "_sess{}.h5".format(i))
            data = h5[sess]
            with h5py.File(new_file, "w") as h5_out:
                for k in data.keys():
                    h5_out.create_dataset(k, data=np.array(data[k]))
                h5.copy("metadata", h5_out, name="metadata")
                h5.copy("__DATA_TYPES__", h5_out, name="__DATA_TYPES__")
    if remove_original:
        os.remove(f)

def main(args):
    if not os.path.exists(args.input):
        raise ValueError("Input file/directory does not exist: {}".format(input))
    
    files = get_files(args.input)

    for f in files:
        split_session(f, args.remove_original)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split h5 files containing multiple sessions into separate files')
    parser.add_argument('-i', '--input', type=str, help='Input file or directory', required=True)
    parser.add_argument('-r', '--remove-original', action='store_true', help='Remove original file')
    args = parser.parse_args()
    main(args)
