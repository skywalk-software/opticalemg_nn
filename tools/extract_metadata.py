import argparse
import os
from tqdm import tqdm
import h5py
import yaml
from datetime import datetime as dt

def is_h5(f):
    return f.endswith(".h5")

def get_input():
    return input("Overwrite? [Y/N/all/none] ")

def process_file(in_file, out_file, clean):
    with h5py.File(in_file, "r") as h5:
        ## What to do with __DATATYPES__?
        ## Want to drop this data from the h5 file too? 
        metadata = list(h5['metadata'][()])
        metadata = [m.decode('utf-8') for m in metadata]
        metadata = {
            "trial_type": metadata[0],
            "user_id": metadata[1],
            "date": str(dt.strptime(metadata[2], '%Y-%m-%dT%H-%M-%S').date()),
            "time": str(dt.strptime(metadata[2], '%Y-%m-%dT%H-%M-%S').time()),
            "firmware_version": metadata[3],
            "hand": metadata[4],
            "notes": metadata[5],
        }
        with open(out_file, "w") as of:
            yaml.safe_dump(metadata, of)
    if clean:
        with h5py.File(in_file, "a") as h5:
            if "metadata" in h5:
                del h5["metadata"]
            if "__DATATYPES__" in h5:
                del h5["__DATATYPES__"]


def load_files(input):
    if not os.path.exists(input):
        raise ValueError("Input file/directory does not exist: {}".format(input))
    if os.path.isdir(input):
        files = os.listdir(input)
    else:
        files = [os.path.abspath(input)]
    files = [f for f in files if is_h5(f)]
    if os.path.isdir(input):
        in_files = [os.path.join(input, f) for f in files]
    else:
        in_files = [input]
    in_files = [os.path.abspath(f) for f in in_files]
    out_files = [f.replace("h5", "yaml") for f in in_files]
    return in_files, out_files

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
    parser.add_argument('-cdf', '--clean-data-files', action='store_true', help='Remove metadata from h5 data files after extraction')
    args = parser.parse_args()
    main(args)
