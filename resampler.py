from osgeo import gdal
from osgeo import gdalconst
from scipy.interpolate import griddata as interpolate
import rhealpix_dggs.dggs as dggs
import numpy as np
import h5py
import sys

def cell_name(cell):
    return '/'.join(str(cell))

def fromFile(filename, max_resolution, hdf5_file, band_num):
    """ Given a GeoTIFF filename, returns an array of dicts (one for each band) of the form:
        {cell : value}, for rHEALPix cells of the given resolution (we normally use 13, slightly larger than a landsat pixel)
        each cell's value is interpolated from nearby pixels, and missing pixels are ignored
    """
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
            outer_cell = rddgs.cell_from_point(outer_res, (transform[0], transform[3]), plane=False)
            break

    print("Processing band ", band_num, "/", numBands, "...")
    outer_group = hdf5_file.create_group(cell_name(outer_cell))
    band = dataset.GetRasterBand(band_num)
    missing_val = band.GetNoDataValue()
    masked_data = np.ma.masked_equal(band.ReadAsArray(), missing_val)
    outer_pixel = outer_group.create_dataset('pixel', (1,))
    outer_pixel[0] = masked_data.mean()
    for resolution in range(outer_res + 1, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")
        cells = rddgs.cells_from_region(resolution, (transform[0], transform[3]), (transform[0] + width * transform[1], transform[3] + height * transform[5]), plane=False)
        for cells2 in cells:
            for cell in cells2:
                north_west, _, south_east, _ = cell.vertices(plane=False)
                left, top = north_west
                right, bottom = south_east
                left   = max(0, int((left   - transform[0]) / transform[1]))
                right  = max(0, int((right  - transform[0]) / transform[1]))
                top    = max(0, int((top    - transform[3]) / transform[5]))
                bottom = max(0, int((bottom - transform[3]) / transform[5]))
                value = masked_data[left:right,top:bottom].mean()
                if value is not np.ma.masked:
                    group = hdf5_file.create_group(cell_name(cell))
                    pixel = group.create_dataset('pixel', (1,))
                    pixel[0] = value

def run_code(tif_name, hdf5_name, max_res, band_num):
    hdf5_file = h5py.File(hdf5_name, "w")
    try:
        fromFile(tif_name, max_res, hdf5_file, band_num)
    finally:
        hdf5_file.close()
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("-------------------------------------------------------------")
        print("Usage: resampler.py input.tif output.hdf5 max_resolution band")
        print("-------------------------------------------------------------")
        print("Example: resampler.py test.tif result.hdf5 8 2")
    else:
        run_code(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))