import argparse
import warnings
import os
from tqdm import tqdm
import h5py
import yaml

def is_h5(f):
    return f.endswith(".h5")

def process_files(in_files, out_files, clean):
    for i, f in tqdm(enumerate(in_files), total=len(in_files)):
        print("Processing {}".format(f))
        with h5py.File(f, "r") as h5:

            ## Best way to structure metadata in yaml?
            ## How about __DATATYPES__?

            metadata = list(h5['metadata'][()])
            metadata = [m.decode('utf-8') for m in metadata]
            with open(out_files[i], "w") as of:
                yaml.safe_dump(metadata, of)
        if clean:
            with h5py.File(f, "a") as h5:
                del h5['metadata']
                del h5['__DATATYPES__']


def main(args):
    if not os.path.exists(args.input):
        raise ValueError("Input file/directory does not exist: {}".format(args.input))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        warnings.warn("Output directory already exists and may contain files from previous runs")
        resp = input("Continue? [y/N] ")
        if resp.lower() != "y":
            raise ValueError("Aborted")
    
    if os.path.isdir(args.input):
        files = os.listdir(args.input)
    else:
        files = [args.input]

    for f in files:
        if not is_h5(f):
            warnings.warn("Skipping non-h5 file found in input directory: {}".format(f))
            files.remove(f)
    
    if len(files) == 0:
        print("No h5 files found... exiting")
        exit(0)

    out_files = [f.replace("h5", "yaml") for f in files]
    in_files = [os.path.join(args.input, f) for f in files]
    out_files = [os.path.join(args.output, f) for f in out_files]

    for i, of in enumerate(out_files):
        if os.path.exists(of):
            warnings.warn("Output file already exists: {}".format(of))
            resp = input("Overwrite? [y/N] ")
            if resp.lower() != "y":
                print("Skipping {}".format(of))
                out_files.remove(of)
                in_files.remove(in_files[i])

    process_files(in_files, out_files, args.clean_data_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove metadata from h5 data file and save as yaml')
    parser.add_argument('-i', '--input', type=str, help='Input file or directory')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('-cdf', '--clean-data-files', action='store_true', help='Remove metadata from h5 data files after extraction')
    args = parser.parse_args()
    main(args)
