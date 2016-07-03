#!/usr/bin/env python3

from osgeo import gdal
from osgeo import gdalconst
import sys

def show_data(tif_name, band_num):
    dataset = gdal.Open(tif_name, gdalconst.GA_ReadOnly)
    band = dataset.GetRasterBand(band_num)
    band_data = band.ReadAsArray()
    plt.imshow(band_data, interpolation='none')
    plt.show()

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

    if len(sys.argv) == 3:
        show_data(sys.argv[1], int(sys.argv[2]))
    else:
        print("-------------------------------------------------------------")
        print("Usage:")
        print("display.py input.tif band")
        print("-------------------------------------------------------------")
