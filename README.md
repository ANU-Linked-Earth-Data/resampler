# DGGS resampler script

This repo has some scripts for messing with satellite data and DGGSes. In
particular, `resampler.py` can take a GeoTIFF and stick it into a custom HDF5
format which partitions data into a different HDF5 dataset for each grid cell.

To run the resampler, do the following:

```
# Make sure you have Python 3 and virtualenv installed first!
virtualenv env -p `which python3`
. env/bin/activate
pip install -r requirements.txt
./resampler.py some-geotiff-you-downloaded.tif result.h5 2 7 5
```

The Python 3 requirement is important! Also, you might get some issues
installing GDAL. If so, try something like:

```
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" \
    GDAL==1.11.2
```

You can replace `1.11.2` with a more recent release in the `1.11.X` series if
one is available.
