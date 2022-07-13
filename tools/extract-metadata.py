import argparse
import os
from tqdm import tqdm
import h5py
import yaml
from datetime import datetime as dt

def is_h5(f):
    return f.endswith(".h5")

def process_files(in_files, out_files, clean):
    for i, f in tqdm(enumerate(in_files), total=len(in_files)):
        try:
            with h5py.File(f, "r") as h5:

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
                with open(out_files[i], "w") as of:
                    yaml.safe_dump(metadata, of)
            if clean:
                with h5py.File(f, "a") as h5:
                    del h5['metadata']
                    del h5['__DATATYPES__']
        except Exception as e:
            print("Error processing {}: {}".format(f, e))
            continue


def main(args):
    if not os.path.exists(args.input):
        raise ValueError("Input file/directory does not exist: {}".format(args.input))
    
    if os.path.isdir(args.input):
        files = os.listdir(args.input)
    else:
        files = [os.path.abspath(args.input)]

    files = [f for f in files if is_h5(f)]
    
    if len(files) == 0:
        print("No h5 files found... exiting")
        exit(0)

    if os.path.isdir(args.input):
        in_files = [os.path.abspath(os.path.join(args.input, f)) for f in files]

    out_files = [f.replace("h5", "yaml") for f in in_files]

    to_drop = []
    for j in range(len(in_files)):
        if os.path.isfile(out_files[j]):
            print("Output file already exists: {}".format(out_files[j]))
            resp = input("Overwrite? [y/N/all] ")
            if resp == "all":
                print("Overwriting all")
                break
            if resp == "N":
                print("Skipping {}".format(out_files[j]))
                to_drop.append((in_files[j], out_files[j]))
    
    for ifl, ofl in to_drop:
        in_files.remove(ifl)
        out_files.remove(ofl)
    
    if len(out_files) == 0:
        print("No output files to write... exiting")
        exit(0)

    process_files(in_files, out_files, args.clean_data_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove metadata from h5 data file and save as yaml')
    parser.add_argument('-i', '--input', type=str, help='Input file or directory', required=True)
    parser.add_argument('-cdf', '--clean-data-files', action='store_true', help='Remove metadata from h5 data files after extraction')
    args = parser.parse_args()
    main(args)
