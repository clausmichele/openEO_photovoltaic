import os
import copy
import openeo
import numpy as np
import geopandas as gpd

source_geometries = "/mnt/CEPH_PROJECTS/sao/openEO_Platform/germany_photovoltaic.shp"

big_pv_geoms = gpd.read_file(source_geometries)

print(f"Number of selected PV Farms: {len(big_pv_geoms)}")

LCLU_folder = "/mnt/CEPH_PROJECTS/sao/openEO_Platform/lclu/germany/"

# Apply a buffer of 20 meters around our PV farm polygon to get some pixels around it
big_pv_geoms_32632 = big_pv_geoms.to_crs(32632)
big_pv_geoms_32632_buffer_20 = copy.deepcopy(big_pv_geoms_32632)
big_pv_geoms_32632_buffer_20["geometry"] = [i.buffer(20) for i in big_pv_geoms_32632_buffer_20.geometry]
big_pv_geoms_buffer = big_pv_geoms_32632_buffer_20.to_crs(4326)

## Download LCLU Maps
for i, geom in big_pv_geoms_buffer.iterrows():
    print(i)
    if os.path.exists(f"{LCLU_folder}lclu_2021_{i}.nc"):
        print('Already downloaded')
        continue
    conn = openeo.connect("openeo.cloud").authenticate_oidc()
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
        lclu_cube.download(f"{LCLU_folder}lclu_2021_{i}.nc")
    except:
        print(f'{i} did not downloaded')
        continue