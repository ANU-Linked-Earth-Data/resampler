#!/usr/bin/env python3

import h5py
import sys

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

def show_data(hdf5_name, cell):
    hdf5_file = h5py.File(hdf5_name, "r")
    try:
        data = hdf5_file[cell_name(cell) + "/data"]
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

if __name__ == "__main__":
    # For some reason, I need to set the backend to get my system to display
    # anything. Otherwise, plt.show() just exits silently.
    plt = init_pyplot()

    if len(sys.argv) == 2:
        show_tree(sys.argv[1])
    elif len(sys.argv) == 3:
        show_data(*sys.argv[1:])
    else:
        print("-------------------------------------------------------------")
        print("Usage:")
        print("display.py result.hdf5")
        print("display.py result.hdf5 R7852")
        print("-------------------------------------------------------------")
