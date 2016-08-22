#!/usr/bin/env python3

import h5py
from io import BytesIO
from scipy.ndimage import imread

from argparse import ArgumentParser

def cell_name(cell):
    return '/'.join(str(cell))

def show_tree(hdf5_name):
    hdf5_file = h5py.File(hdf5_name, "r")
    try:
        hdf5_file.visit(lambda name: print(name))
        for k, v in hdf5_file.attrs.items():
            print(k + ": " + str(v))
    finally:
        hdf5_file.close()

def show_data(hdf5_name, cell, band):
    with h5py.File(hdf5_name, "r") as hdf5_file:
        ds_name = '%s/png_band_%i' % (cell_name(cell), band)
        # .data is possibly unsafe? Pretty sure endianness and layout don't
        # matter for *this* dataset, but could be wrong.
        png_data = BytesIO(hdf5_file[ds_name].value.data)
        data = imread(png_data)
        plt.imshow(data, interpolation='none')
        plt.show()

def init_pyplot():
    import matplotlib as mpl
    valid_bk = mpl.rcsetup.interactive_bk
    if mpl.rcParams['backend'] not in valid_bk:
        # Fall back to Tk (might require extra modules)
        mpl.rcParams['backend'] = "TkAgg"

    import matplotlib.pyplot as plt
    return plt

parser = ArgumentParser()
parser.add_argument('input', type=str, help='path to input HDF5 file')
parser.add_argument('--cell', '-c', type=str, default=None, help='rHEALPix cell to display')
parser.add_argument('--band', '-b', type=int, dest='band', default=2, help='Landsat band to display')

if __name__ == "__main__":
    # For some reason, I need to set the backend to get my system to display
    # anything. Otherwise, plt.show() just exits silently.
    plt = init_pyplot()
    args = parser.parse_args()

    if not args.cell:
        show_tree(args.input)
    else:
        show_data(args.input, args.cell, args.band)
