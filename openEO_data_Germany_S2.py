import os
import copy
import numpy as np
import geopandas as gpd
import openeo
from openeo.processes import lte
import xarray as xr
from scipy.stats import mode

big_pv_geoms = gpd.read_file('/mnt/CEPH_PROJECTS/sao/openEO_Platform/germany_photovoltaic.shp')

print(f"Number of selected PV Farms: {len(big_pv_geoms)}")

# Apply a buffer of 20 meters around our PV farm polygon to get some pixels around it
big_pv_geoms_32632 = big_pv_geoms.to_crs(32632)
big_pv_geoms_32632_buffer_20 = copy.deepcopy(big_pv_geoms_32632)
big_pv_geoms_32632_buffer_20["geometry"] = [i.buffer(20) for i in big_pv_geoms_32632_buffer_20.geometry]
big_pv_geoms_buffer = big_pv_geoms_32632_buffer_20.to_crs(4326)

## Check the already downloaded LCLU Maps
# lclu_netcds = []
# for i, geom in big_pv_geoms_buffer.iterrows():
#     if os.path.exists(f"/mnt/CEPH_PROJECTS/sao/openEO_Platform/lclu/germany/lclu_2021_{i}.nc"):
#         lclu_netcds.append(i)

LCLU_valid_idx = []
LCLU_values = []
lclu_dic = {10:"Tree cover", 20:"Shrubland", 30:"Grassland",
            40:"Cropland", 50:"Built-up", 60:"Bare / sparse vegetation",
            70:"Snow and Ice", 80:"Permanent water bodies", 90:"Herbaceous Wetland",
            95:"Mangrove" , 100:"Moss and lichen"}
if not os.path.exists("LCLU_valid_idx.txt"):
    for i, geom in big_pv_geoms.iterrows():
        try:
            data = xr.open_dataset(f"/mnt/CEPH_PROJECTS/sao/openEO_Platform/lclu/germany/lclu_2021_{i}.nc",decode_coords="all")
            geodf = gpd.GeoDataFrame(geometry=[geom["geometry"]],crs="EPSG:4326")
            clipped = data.rio.clip(geodf.geometry.values, geodf.crs, drop=False, invert=False)
            arr = clipped.MAP.values.ravel()
            result = mode(arr, nan_policy = 'omit')
            mode_value = int(result.mode)

            clipped_inv = data.rio.clip(geodf.geometry.values, geodf.crs, drop=False, invert=True)
            arr_inv = clipped_inv.MAP.values.ravel()
            result_inv = mode(arr_inv, nan_policy = 'omit')
            mode_value_inv = int(result_inv.mode)

            # If the area is marked as built-up but all around it it's covered by a natural/vegetated area, we consider it valid
            if mode_value == 50:
                if mode_value_inv in [10,20,30,40,60,90,95,100]:  
                    LCLU_values.append(mode_value_inv)
                    LCLU_valid_idx.append(i)
                else:
                    LCLU_values.append(mode_value)
            elif mode_value == 80: #We don't consider plants 
                LCLU_values.append(mode_value)
            else:
                LCLU_values.append(mode_value)
                LCLU_valid_idx.append(i)
        except Exception as e:
            LCLU_values.append(np.nan)
            print(e)
            print(i)
        
    LCLU_valid_idx_str = [str(x) + "," for x in LCLU_valid_idx]
    with open("LCLU_valid_idx.txt","w") as f:
        f.writelines(LCLU_valid_idx_str)
else:
    LCLU_valid_idx = sorted(np.fromfile("LCLU_valid_idx.txt",dtype=np.uint16,sep=","))


years = [2022]
# properties = {"eo:cloud_cover": lambda x: x.lte(65)}
for y in years:
    for i, geom in big_pv_geoms_buffer.iterrows():
        if i in LCLU_valid_idx:
            if os.path.exists(f"/mnt/CEPH_PROJECTS/sao/openEO_Platform/s2/germany/s2_{y}_{i}.nc"):
                print(f'{y} S2 data already downloaded')
                continue
            else:
                conn = openeo.connect("openeo.cloud").authenticate_oidc()
                aoi = geom["geometry"].bounds
                collection = "SENTINEL2_L2A"
                spatial_extent = {"west": aoi[0],
                                  "east": aoi[2],
                                  "south": aoi[1],
                                  "north": aoi[3]}
                temporal_extent = [f"{y}-01-01",f"{y}-12-31"]
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","SCL"]
                s2_cube = conn.load_collection(
                    collection,
                    spatial_extent = spatial_extent,
                    temporal_extent = temporal_extent,
                    bands = s2_bands,
                )
                try:
                    s2_cube.download(f"/mnt/CEPH_PROJECTS/sao/openEO_Platform/s2/germany/s2_{y}_{i}.nc")
                except:
                    print(f'{i} failed to download')
                    continue
