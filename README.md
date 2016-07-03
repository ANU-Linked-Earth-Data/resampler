# DGGS resampler script

This repo has some scripts for messing with satellite data and DGGSes. In
particular, `resampler.py` can take a GeoTIFF and stick it into a custom HDF5
format which partitions data into a different HDF5 dataset for each grid cell.

Install procedure is the same as that for every other Python script: `virtualenv
env && . env/bin/activate && pip install -r requirements.txt`.

You might have some issues installing GDAL. If so, try something like:

```sh
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" \
    GDAL==2.1.0
```

You can replace `2.1.0` with a more recent release in the `2.X.X` series if one
is available.

For whatever reason, the `display` script didn't work for me initially (in a
virtualenv on Ubuntu 16.04). I fixed the issue by doing `sudo apt install
tk-dev` and updating `display.py` to fall back to the Tk matplotlib renderer if
no other interactive renderer is enabled.

## Merging files

The `gdal_merge.py` script in this directory, which was copied verbatim from
GDAL, can combine arbitrarily many TIFF files into one. To save you the trouble
of running it manually, there's a `merge_all.py` script which can merge Landsat
tiles with the same observation time into one big file. Just give it all your
Landsat files (regardless of what timestamps they have) and it will do the heavy
lifting. Example:

```
(env) $ ls data/ | cat
LS8_OLI_TIRS_NBAR_148_-035_2013-05-27T23-58-20.tif
LS8_OLI_TIRS_NBAR_148_-035_2014-01-22T23-57-25.tif
LS8_OLI_TIRS_NBAR_148_-035_2014-05-30T23-55-52.tif
... snip ...
LS8_OLI_TIRS_NBAR_149_-036_2016-05-19T23-55-52.tif
(env) $ ./merge_all.py --outdir out/ data/*
```

After the script finishes, the merged tiles should be in `out/`.
