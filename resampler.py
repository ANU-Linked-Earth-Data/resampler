#!/usr/bin/env python3

from osgeo import gdal, gdalconst, osr
import pytz
from rhealpix_dggs import dggs
import numpy as np
import h5py

from argparse import ArgumentParser
from itertools import chain
from os.path import basename
import re
from time import time

# For parsing AGDC filenames
AGDC_RE = re.compile(
    r'^(?P<sat_id>[^_]+)_(?P<sensor_id>ETM|OLI_TIRS)_(?P<prod_code>[^_]+)_'
    r'(?P<lon>[^_]+)_(?P<lat>[^_]+)_(?P<year>\d+)-(?P<month>\d+)-'
    r'(?P<day>\d+)T(?P<hour>\d+)-(?P<minute>\d+)-(?P<second>\d+(\.\d+)?)'
    r'\.tif$'
)

def cell_name(cell):
    return '/'.join(str(cell))

def parse_agdc_fn(fn):
    """Parses a filename in the format used by the AGDC Landsat archive. For
    example, ``LS7_ETM_NBAR_149_-036_2012-02-10T23-50-47.650696.tif`` is a
    Landsat 7 observation in GeoTIFF format taken on Februrary 10 2012 at
    11:50pm (amongst other things).

    >>> fn = 'LS7_ETM_NBAR_149_-036_2012-02-10T23-50-47.650696.tif'
    >>> sorted_results = sorted(parse_agdc_fn(fn).items())
    >>> print('\\n'.join('{}: {}'.format(k, v) for k, v in sorted_results))
    datetime: 2012-02-10 23:50:47.650696+00:00
    lat: -36.0
    lon: 149.0
    prod_code: NBAR
    sat_id: LS7
    sensor_id: ETM

    :param string fn: Filename for observation.
    :return: Dictionary of extracted metdata from filename.
    """
    match = AGDC_RE.match(fn)
    if match is None:
        raise ValueError('Invalid AGDC filename: "{}"'.format(fn))
    info = match.groupdict()
    raw_sec = float(info['second'])
    int_sec = int(raw_sec)
    microsecond = int(1e6 * (raw_sec % 1))
    dt = pytz.datetime.datetime(
        year=int(info['year']), month=int(info['month']), day=int(info['day']),
        hour=int(info['hour']), minute=int(info['minute']),
        second=int_sec, microsecond=microsecond, tzinfo=pytz.utc
    )
    rv = {
        'lat': float(info['lat']), 'lon': float(info['lon']), 'datetime': dt,
        'prod_code': info['prod_code'], 'sensor_id': info['sensor_id'],
        'sat_id': info['sat_id']
    }
    return rv


def add_meta(metadata, group):
    """Add metadata, probably AGDC-related (e.g. acquisition time, satellite
    ID), to an HDF5 group."""
    for k, v in metadata.items():
        if isinstance(v, pytz.datetime.datetime):
            enc_v = v.isoformat()
        else:
            # Hope for the best!
            enc_v = v
        group.attrs[k] = enc_v
    return group


def pixel_to_long_lat(geotransform, dataset_projection, col, row):
    """ Given a pixel position as a column/row, calculates its position in the dataset's reference system,
        then converts it to a latitude and longitude in the WGS_84 system """
    tx = osr.CoordinateTransformation (dataset_projection, wgs_84_projection)
    lon, lat, height = tx.TransformPoint(*gdal.ApplyGeoTransform(geotransform, col, row))
    return lon, lat


rhealpix_proj4_string = "+proj=rhealpix +I +lon_0=0 +a=1 +ellps=WGS84 +npole=0 +spole=0 +wktext"

rhealpix_projection = osr.SpatialReference()
rhealpix_projection.ImportFromProj4(rhealpix_proj4_string)

wgs_84_projection = osr.SpatialReference()
wgs_84_projection.ImportFromEPSG(4326)

def reproject_dataset (dataset, dataset_projection, cell, resolution_gap):
    """ Based on https://jgomezdans.github.io/gdal_notes/reprojection.html """
    data_to_rhealpix = osr.CoordinateTransformation (dataset_projection, rhealpix_projection)
    lonlat_to_rhealpix = osr.CoordinateTransformation (wgs_84_projection, rhealpix_projection)
    
    geo_t = dataset.GetGeoTransform()
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    (ulx, uly, ulz ) = data_to_rhealpix.TransformPoint( geo_t[0], geo_t[3])
    (lrx, lry, lrz ) = data_to_rhealpix.TransformPoint( geo_t[0] + geo_t[1]*x_size,geo_t[3] + geo_t[5]*y_size )

    # Calculate the new geotransform
    north_west, _, south_east, _ = cell.vertices(plane=False)
    left, top = north_west
    right, bottom = south_east
    left, top, _ = lonlat_to_rhealpix.TransformPoint(left, top)
    right, bottom, _ = lonlat_to_rhealpix.TransformPoint(right, bottom)
    num_pixels = 3 ** resolution_gap
    new_geo = ( left, (right - left) / num_pixels, 0, \
                top, 0, (bottom - top) / num_pixels )
    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', num_pixels, num_pixels, dataset.RasterCount, dataset.GetRasterBand(1).DataType)
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(rhealpix_projection.ExportToWkt())

    # Perform the projection/resampling
    error_code = gdal.ReprojectImage(dataset, dest, dataset_projection.ExportToWkt(), rhealpix_projection.ExportToWkt(), gdal.GRA_Bilinear)
    assert error_code == 0, "Reprojection failed"

    return dest
    
def open_dataset(filename):
    """ Reads a geotiff or a HDF4 file and returns a gdal dataset """
    if filename.split(".")[-1] == "tif":
        return gdal.Open(filename, gdalconst.GA_ReadOnly)
    elif filename.split(".")[-1] == "hdf":
        dataset = gdal.Open(filename, gdalconst.GA_ReadOnly) #Yay gdal can open MODIS hdf files! :D
        ### But it doesn't read in the georeferencing system properly ...
        
        from pyhdf.SD import SD, SDC
        hdf = SD(filename, SDC.READ)
        latitudes = hdf.select('latitude')[:]
        longitudes = hdf.select('longitude')[:]
        
        left = longitudes[0]
        top = latitudes[0]
        x_spacing = np.mean([longitudes[i+1] - longitudes[i] for i in range(len(longitudes)-1)])
        y_spacing = np.mean([latitudes[i+1] - latitudes[i] for i in range(len(latitudes)-1)])
        
        geotransform = ( left, x_spacing, 0, \
                top, 0, y_spacing )
        
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(wgs_84_projection.ExportToWkt()) # This will do for now
        
        return dataset
    else:
        assert False, "Invalid file extension " + filename.split(".")[-1:] + ", expected 'tif' or 'hdf'"
        
        
def from_file(filename, dataset, hdf5_file, max_resolution, resolution_gap):
    """ Converts a gdal dataset into a hdf5 rhealpix file """
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    rddgs = dggs.RHEALPixDGGS()

    dataset_projection = osr.SpatialReference()
    error_code = dataset_projection.ImportFromWkt(dataset.GetProjection())
    assert error_code == 0, "Dataset doesn't have a projection"

    try:
        tif_meta = parse_agdc_fn(basename(filename))
        add_meta(tif_meta, hdf5_file)
    except ValueError:
        print("Can't read metadata from filename. Is it in the AGDC format?")
        tif_meta = None

    upper_left = pixel_to_long_lat(geotransform, dataset_projection, 0, 0)
    lower_right = pixel_to_long_lat(geotransform, dataset_projection, width, height)

    try:
        bounding_cell = rddgs.cell_from_region(upper_left, lower_right, plane=False)
        outer_res = bounding_cell.resolution
    except AttributeError: # dggs library produces this error, maybe when even top-level cells are too small?
        outer_res = 0

    for resolution in range(outer_res, max_resolution + 1):
        print("Processing resolution ", resolution, "/", max_resolution, "...")

        cells = rddgs.cells_from_region(
            resolution, upper_left, lower_right, plane=False
        )
        for cell in chain(*cells):
            north_west, north_east, south_east, south_west = cell.vertices(plane=False)
            if cell.region() != "equatorial":
                continue # Yucky polar cells, ignore for now, maybe fix later

            data = reproject_dataset(dataset, dataset_projection, cell, resolution_gap).ReadAsArray()
            if not np.any(data):
                continue
                
            pixel_value = np.array([(np.mean(x[np.nonzero(x)]) if np.any(x[np.nonzero(x)]) else 0) for x in data])
            
            # Write the HDF5 group. This is much faster than writing inline,
            # and lets us use numba.
            group = hdf5_file.create_group(cell_name(cell))
            if tif_meta is not None:
                add_meta(tif_meta, group)
            group.attrs['bounds'] = np.array([
                north_west, north_east, south_east, south_west, north_west
            ])
            group.attrs['centre'] = np.array(cell.centroid(plane=False))
            group.attrs['missing_value'] = 0 # Value used by gdal.ReprojectImage()
            group['pixel'] = pixel_value
            group.create_dataset('data', data=data, compression='szip')


parser = ArgumentParser()
parser.add_argument('input', type=str, help='path to input GeoTIFF or MODIS HDF4 file')
parser.add_argument('output', type=str, help='path to output HDF5 file')
parser.add_argument(
    '--max-res', type=int, dest='max_res', default=6,
    help='maximum DGGS depth to resample at'
)
parser.add_argument(
    '--res-gap', type=int, dest='res_gap', default=5,
    help='number of DGGS levels to go down when generating tile data'
)

if __name__ == "__main__":
    args = parser.parse_args()
    print('Reading from %s and writing to %s' % (args.input, args.output))
    print(
        'Resampling to depth %i with gap %i (so %i pixels per tile)'
        % (args.max_res, args.res_gap, 9 ** args.res_gap)
    )

    dataset = open_dataset(args.input)
    
    start_time = time()
    with h5py.File(args.output, "w") as hdf5_file:
        from_file(
            args.input, dataset, hdf5_file, args.max_res, args.res_gap
        )

    elapsed = time() - start_time
    print("Done! Took %.2fs" % elapsed)
