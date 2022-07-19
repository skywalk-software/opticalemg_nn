import argparse
import os
from tqdm import tqdm
import h5py
import yaml
from datetime import datetime as dt
from boltons.fileutils import iter_find_files

notes = [
    'light flexion extension of wrist during events, stable otherwise',
    'mix of open and close other fingers, stable otherwise',
    'idle motion, knocking table, drinking water, adjusting band slightly',
    'realistic HL2 use (longer holds) with rest periods',
    'mix of rotation, flexion, move, open/close other fingers',
    'consistent hand position, no noise',
    'mid-air tap for gaze pointer, with fingers open/close',
    'nothing',
    'guitarhero with rotation, motion, fingers open/close, interspersed with idle motion, picking up objects, resting hand',
    'none',
    'light rotation of wrist in between events',
    'unidirectional motion of hand during hold events, emulating click + drag',
    'just resting + idle motion, adjusting band slightly',
    'mid-air tap and hold for raycast, fingers closed'
]

## *** := notes type not used in previous training script 
trial_type = [
    'flexion_extension',
    'open_close',
    'idle_motion_band_adjust',      # ***
    'realistic',
    'all_mixed',
    'simple',
    'mid_air_tap_gaze_pointer',     # ***
    'nothing',                      # ***   
    'guitarhero_rotation_motion',   # ***
    'none',                         # ***
    'rotation',
    'drag',
    'resting_idle_motion',          # ***
    'mid_air_tap_hold_raycast',     # ***
]

notes_to_trialtype = dict(zip(notes, trial_type))

def is_h5(f):
    return f.endswith(".h5")

def get_input():
    return input("Overwrite? [Y/N/all/none] ")

def process_file(in_file, out_file, clean):
    with h5py.File(in_file, "r") as h5:
        metadata = list(h5['metadata'][()])
        metadata = [m.decode('utf-8') for m in metadata]
        keys = list(h5.keys())
        keys.remove('metadata')
        keys.remove('__DATA_TYPES__')
        metadata = {
            "datacollector_version": metadata[0],
            "user": metadata[1],
            "date": str(dt.strptime(metadata[2], '%Y-%m-%dT%H-%M-%S').date()),
            "time": str(dt.strptime(metadata[2], '%Y-%m-%dT%H-%M-%S').time()),
            "firmware_version": metadata[3],
            "hand": metadata[4],
            "notes": metadata[5],
            "trial_type": notes_to_trialtype[metadata[5]],
            "filename": os.path.basename(in_file).replace(".h5", ""),
            "session_ids": keys,
        }
        with open(out_file, "w") as of:
            yaml.safe_dump(metadata, of)
    if clean:
        with h5py.File(in_file, "a") as h5:
            if "metadata" in h5:
                del h5["metadata"]
            if "__DATATYPES__" in h5:
                del h5["__DATA_TYPES__"]

def load_files(input):
    if not os.path.exists(input):
        raise ValueError("Input file/directory does not exist: {}".format(input))
    if os.path.isdir(input):
        files = iter_find_files(input, patterns="*.h5", include_dirs=True)
        files = [os.path.abspath(f) for f in files]
    else:
        files = [os.path.abspath(input)]
    files = [f for f in files if is_h5(f)]
    out_files = [f.replace("h5", "yaml") for f in files]
    return files, out_files

def check_for_existing(in_files, out_files):
    to_drop = []
    for j in range(len(in_files)):
        if os.path.isfile(out_files[j]):
            print("Output file already exists: {}".format(out_files[j]))
            resp = get_input()
            if resp == "all":
                print("Overwriting all")
                break
            if resp == "none":
                return [], []
            if resp == "N":
                print("Skipping {}".format(out_files[j]))
                to_drop.append((in_files[j], out_files[j]))
    for ifl, ofl in to_drop:
        in_files.remove(ifl)
        out_files.remove(ofl)
    return in_files, out_files


def main(args):

    in_files, out_files = load_files(args.input)

    if len(in_files) != len(out_files):
        raise ValueError("Number of input files ({}) does not \
            match number of output files ({})".format(len(in_files), len(out_files)))
    
    if len(in_files) == 0:
        print("No h5 files found... exiting")
        exit(0)

    in_files, out_files = check_for_existing(in_files, out_files)
    
    if len(in_files) == 0:
        print("No files left to process... exiting")
        exit(0)

    for i, f in tqdm(enumerate(in_files), total=len(in_files)):
        try:
            process_file(in_files[i], out_files[i], args.clean_data_files)
        except Exception as e:
            print("Error processing {}: {}".format(f, e))
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove metadata from h5 data file and save as yaml')
    parser.add_argument('-i', '--input', type=str, help='Input file or directory', required=True)
    parser.add_argument('-c', '--clean-data-files', action='store_true', help='Remove metadata from h5 data files after extraction')
    args = parser.parse_args()
    main(args)
