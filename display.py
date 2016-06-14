#!/usr/bin/env python3

import h5py
import sys
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
    hdf5_file = h5py.File(hdf5_name, "r")
    try:
        data = hdf5_file[cell_name(cell) + "/data"]
        if len(data.shape) == 3:
            data = data[band - 1]
        assert len(data.shape) == 2, "Something is wrong with the dimensions of the data"
        plt.imshow(data, interpolation='none')
        plt.show()
        print(data.shape)
    finally:
        hdf5_file.close()

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
