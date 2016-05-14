from osgeo import gdal
from osgeo import gdalconst
from scipy.interpolate import griddata as interpolate
import rhealpix_dggs.dggs as dggs
import numpy as np


def fromFile(filename, resolution):
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
    cells = rddgs.cells_from_region(resolution, (transform[0], transform[3]), (transform[0] + width * transform[1], transform[3] + height * transform[5]), plane=False)
    cells = [cell for row in cells for cell in row]
    cell_coords = [cell.centroid(plane=False) for cell in cells]
    for band_num in range(1,numBands + 1):
        print("Processing band ", band_num, "/", numBands, "...")
        band = dataset.GetRasterBand(band_num)
        missing_val = band.GetNoDataValue()
        masked_data = np.ma.masked_equal(band.ReadAsArray(), missing_val)
        x_coords = np.ma.array([[transform[0] + (x + 0.5) * transform[1] for x in range(width)] for y in range(height)], mask=masked_data.mask)
        y_coords = np.ma.array([[transform[3] + (y + 0.5) * transform[5] for x in range(width)] for y in range(height)], mask=masked_data.mask)
        cell_values = interpolate((x_coords.compressed(), y_coords.compressed()), masked_data.compressed(), cell_coords)
        result[band_num - 1] = {str(cell) : cell_values[i] for i, cell in enumerate(cells) if not np.isnan(cell_values[i])}
    
    return result
    
    