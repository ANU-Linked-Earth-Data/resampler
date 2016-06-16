#!/usr/bin/env python3

"""Script to merge a bunch of Landsat observations"""

from argparse import ArgumentParser
from os import path, makedirs
from subprocess import run
from sys import stderr

from resampler import parse_agdc_fn


def check_sane(group):
    """Make sure everything in the group has the same metadata, modulo lat and
    lon."""
    attrs = None

    for info in group:
        dup_info = dict(info)

        # Remove lat and lon
        for prohib in ('lat', 'lon'):
            if prohib in dup_info:
                del dup_info[prohib]

        if attrs is None:
            # Use the first file as a reference
            attrs = dup_info
        else:
            # Do the sanity check
            if dup_info.items() != attrs.items():
                msg = "File '{}' doesn't match '{}' in same group".format(
                    attrs, dup_info
                )
                raise ValueError(msg)


def split_batches(filenames):
    """Split input filenames into batches taken at the same time; the batches
    will be merged together. Also asserts that non-location metadata (e.g.
    satellite, product code, etc.) match for each group)"""
    by_time = {}
    for path_name in filenames:
        file_name = path.basename(path_name)
        parsed_fn = parse_agdc_fn(file_name)
        dt = parsed_fn['datetime']
        by_time.setdefault(dt, []).append((path_name, parsed_fn))

    rv = list(by_time.values())

    for group in rv:
        # Will raise exception if group is non-homogeneous
        check_sane(parsed for _, parsed in group)

    return rv


def process_batch(batch, out_dir):
    assert batch, 'batch must not be empty'
    _, all_attrs = batch[0]
    in_paths = list(p for p, _ in batch)

    # Emulate AGDC filename notation for output file
    out_fn = '{sat_id}_{sensor_id}_{prod_code}_-190_-100_{datetime.year}-' \
        '{datetime.month}-{datetime.day}T{datetime.hour}-{datetime.minute}-' \
        '{datetime.second}.tif'.format(**all_attrs)
    out_path = path.join(out_dir, out_fn)

    # Warn the user (but don't bother stopping) if we're about to write over
    # something
    print('Writing batch of size %i to %s' % (len(batch), out_fn))
    print('Batch files:\n' + '\n'.join('\t' + p for p in in_paths))
    if path.exists(out_path):
        print('Warning: overwriting existing file', file=stderr)

    # Merge all of the TIFFs with external script
    run(['python', './gdal_merge.py', '-o', out_path] + in_paths, check=True)


parser = ArgumentParser()
parser.add_argument(
    '--outdir', type=str, default='out', help='output directory'
)
parser.add_argument(
    'files', type=str, nargs='+', help='input AGDC tiles'
)


if __name__ == '__main__':
    args = parser.parse_args()
    makedirs(args.outdir, exist_ok=True)
    batches = split_batches(args.files)
    for idx, batch in enumerate(batches):
        print('Batch %i/%i' % (idx + 1, len(batches)))
        process_batch(batch, args.outdir)
