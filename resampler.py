from osgeo import gdal
from osgeo import gdalconst
from scipy.interpolate import griddata as interpolate
import rhealpix_dggs.dggs as dggs
import numpy as np
import h5py
import sys

def cell_name(cell):
    return '/'.join(str(cell))

def fromFile(filename, hdf5_file, band_num, max_resolution, resolution_gap):
    """ Reads a geotiff file and converts the data into a hdf5 rhealpix file """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    numBands = dataset.RasterCount
    transform = dataset.GetGeoTransform()
    result = [None]*numBands
    rddgs = dggs.RHEALPixDGGS()
    for resolution in range(0,20):
        cells = rddgs.cells_from_region(resolution, (transform[0], transform[3]), (transform[0] + width * transform[1], transform[3] + height * transform[5]), plane=False)
        if len(cells) > 1:
            outer_res = resolution - 1
            break

    print("Processing band ", band_num, "/", numBands, "...")
    band = dataset.GetRasterBand(band_num)
    missing_val = band.GetNoDataValue()
    masked_data = np.ma.masked_equal(band.ReadAsArray(), missing_val)
    for resolution in range(outer_res, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")
        cells = rddgs.cells_from_region(resolution, (transform[0], transform[3]), (transform[0] + width * transform[1], transform[3] + height * transform[5]), plane=False)
        for cells2 in cells:
            for cell in cells2:
                north_west, _, south_east, _ = cell.vertices(plane=False)
                left, top = north_west
                right, bottom = south_east
                
                left   = int((left   - transform[0]) / transform[1])
                right  = int((right  - transform[0]) / transform[1])
                top    = int((top    - transform[3]) / transform[5])
                bottom = int((bottom - transform[3]) / transform[5])
                
                l = max(0, left)
                r = max(0, right)
                t = max(0, top)
                b = max(0, bottom)
                
                value = masked_data[l:r+1,t:b+1].mean()
                if value is not np.ma.masked:
                    group = hdf5_file.create_group(cell_name(cell))
                    pixel = group.create_dataset('pixel', (1,))
                    pixel[0] = value
                    data = group.create_dataset('data', (3 ** resolution_gap, 3 ** resolution_gap))
                    w = (right - left) / (3 ** resolution_gap)
                    h = (bottom - top) / (3 ** resolution_gap)
                    for x in range(3 ** resolution_gap):
                        for y in range(3 ** resolution_gap):
                            l = max(0, int(left+w*x))
                            r = max(0, int(left+w*(x+1)))
                            t = max(0, int(top+h*y))
                            b = max(0, int(top+h*(y+1)))
                            if l != r and t != b:
                                value = masked_data[l:r+1,t:b+1].mean()
                                if value is not np.ma.masked:
                                    data[x,y] = value
                                else:
                                    data[x,y] = 0
                             

def run_code(tif_name, hdf5_name, band_num, max_res, res_gap):
    hdf5_file = h5py.File(hdf5_name, "w")
    try:
        fromFile(tif_name, hdf5_file, int(band_num), int(max_res), int(res_gap))
    finally:
        hdf5_file.close()
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("-------------------------------------------------------------")
        print("Usage: resampler.py input.tif output.hdf5 band max_resolution resolution_gap")
        print("-------------------------------------------------------------")
        print("Example: resampler.py test.tif result.hdf5 2 7 5")
    else:
        run_code(*sys.argv[1:])