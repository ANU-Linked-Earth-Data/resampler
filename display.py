#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import sys

def cell_name(cell):
    return '/'.join(str(cell))

def show_tree(hdf5_name):
    hdf5_file = h5py.File(hdf5_name, "r")
    try:
        hdf5_file.visit(lambda name: print(name))   
    finally:
        hdf5_file.close()

def show_data(hdf5_name, cell):
    hdf5_file = h5py.File(hdf5_name, "r")
    try:
        data = hdf5_file[cell_name(cell) + "/data"]
        plt.imshow(data)
        plt.show()
    finally:
        hdf5_file.close()

if __name__ == "__main__":
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
