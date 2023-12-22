import os
import copy
import numpy as np
import geopandas as gpd
import leafmap
import openeo
from openeo.processes import lte
import xarray as xr
import matplotlib.pyplot as plt

big_pv_geoms = gpd.read_file('/mnt/CEPH_PROJECTS/sao/openEO_Platform/germany_photovoltaic.shp')

print(f"Number of selected PV Farms: {len(big_pv_geoms)}")

# Apply a buffer of 20 meters around our PV farm polygon to get some pixels around it
big_pv_geoms_32632 = big_pv_geoms.to_crs(32632)
big_pv_geoms_32632_buffer_20 = copy.deepcopy(big_pv_geoms_32632)
big_pv_geoms_32632_buffer_20["geometry"] = [i.buffer(20) for i in big_pv_geoms_32632_buffer_20.geometry]
big_pv_geoms_buffer = big_pv_geoms_32632_buffer_20.to_crs(4326)

## Download LCLU Maps
for i, geom in big_pv_geoms_buffer.iterrows():
    conn = openeo.connect("openeo.cloud").authenticate_oidc()
    print(i)
    if os.path.exists(f"/mnt/CEPH_PROJECTS/sao/openEO_Platform/lclu/germany/lclu_2021_{i}.nc"):
        print('Already downloaded')
        continue
    
    aoi = geom["geometry"].bounds
    collection = "ESA_WORLDCOVER_10M_2021_V2"
    
    spatial_extent = {"west": aoi[0],
                      "east": aoi[2],
                      "south": aoi[1],
                      "north": aoi[3]}
    
    lclu_bands = ["MAP"]
    
    lclu_cube = conn.load_collection(
        collection,
        spatial_extent = spatial_extent,
        bands = lclu_bands,
    )
    
    try:
        lclu_cube.download(f"/mnt/CEPH_PROJECTS/sao/openEO_Platform/lclu/germany/lclu_2021_{i}.nc")
    except:
        print(f'{i} did not downloaded')
        continue
