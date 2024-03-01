import os
import copy
import numpy as np
import geopandas as gpd
import openeo
from openeo.processes import lte
import xarray as xr
from scipy.stats import mode
from time import time

big_pv_geoms = gpd.read_file('/mnt/CEPH_PROJECTS/sao/openEO_Platform/germany_photovoltaic.shp')

print(f"Number of selected PV Farms: {len(big_pv_geoms)}")

big_pv_geoms_32632 = big_pv_geoms.to_crs(32632)
big_pv_geoms_32632_buffer = copy.deepcopy(big_pv_geoms_32632)

# Check the previously downloaded land cover/land use files over the PV farms geometries

LCLU_folder = "/mnt/CEPH_PROJECTS/sao/openEO_Platform/lclu/germany/"
LCLU_valid_idx = []
LCLU_values = []
lclu_dic = {10:"Tree cover", 20:"Shrubland", 30:"Grassland",
            40:"Cropland", 50:"Built-up", 60:"Bare / sparse vegetation",
            70:"Snow and Ice", 80:"Permanent water bodies", 90:"Herbaceous Wetland",
            95:"Mangrove" , 100:"Moss and lichen"}

if not os.path.exists("./aux_files/LCLU_valid_idx.txt"):
    for i, geom in big_pv_geoms.iterrows():
        try:
            data = xr.open_dataset(f"{LCLU_folder}lclu_2021_{i}.nc",decode_coords="all")
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
            elif mode_value == 80: # We don't consider plants over water
                LCLU_values.append(mode_value)
            else:
                LCLU_values.append(mode_value)
                LCLU_valid_idx.append(i)
        except Exception as e:
            LCLU_values.append(np.nan)
            print(e)
            print(i)
        
    LCLU_valid_idx_str = [str(x) + "," for x in LCLU_valid_idx]
    with open("./aux_files/LCLU_valid_idx.txt","w") as f:
        f.writelines(LCLU_valid_idx_str)
else:
    LCLU_valid_idx = sorted(np.fromfile("./aux_files/LCLU_valid_idx.txt",dtype=np.uint16,sep=","))

# The folder where we download the data from openEO
S2_download_folder = "/mnt/CEPH_PROJECTS/sao/openEO_Platform/s2/germany/patches/"

# The years for which we want to download Sentinel-2 features for the selected geometries
years = [2022]

# The could cover used to filter out many cloudy scenes to reduce the computation time
properties = {"eo:cloud_cover": lambda x: x.lte(65)}

# The minimum size (in meters) of the netCDF file we would like to get as output for each geometry
min_patch_size = 5120

# List of dictionaries containing the indexes of the geometries contained by that request,
#the aoi to use in the openeEO graph and the path where to download the netCDF data
geoms_list_dict = []
# List of indices already covered by another download that we will skip
idx_to_skip = []

for y in years:
    for i, geom in big_pv_geoms_32632.iterrows():
        if i in idx_to_skip:
            continue
        if i in LCLU_valid_idx:
            # Apply a buffer of around our PV farm polygon
            geom_bounds = geom["geometry"].bounds
            lon_size = (geom_bounds[2] - geom_bounds[0]) # Meters
            lat_size = (geom_bounds[3] - geom_bounds[1]) # Meters

            lon_to_buffer = np.abs(np.ceil(min_patch_size - lon_size))
            lat_to_buffer = np.abs(np.ceil(min_patch_size - lat_size))
    
            buffer = np.max([lon_to_buffer,lat_to_buffer])/2

            buffered_geom = geom["geometry"].buffer(buffer)

            big_pv_geoms_32632_buffer.loc[[i], "geometry"] = gpd.GeoSeries([buffered_geom])

            if np.sum(big_pv_geoms_32632.intersects(buffered_geom,align=False))>1:
                # We need to use the bounds from all the geometries inside the buffered AOI
                # We will skip the download for the other geometry that we include here
                inside_geoms_idx = []
                intersects = big_pv_geoms_32632[big_pv_geoms_32632.intersects(buffered_geom,align=False)]

                aoi = (np.min([intersects.total_bounds[0],buffered_geom.bounds[0]]),
                       np.min([intersects.total_bounds[1],buffered_geom.bounds[1]]),
                       np.max([intersects.total_bounds[2],buffered_geom.bounds[2]]),
                       np.max([intersects.total_bounds[3],buffered_geom.bounds[3]]))

                inside_geoms_idx = list(intersects.index.values)
                inside_geoms_idx_str = [str(x) for x in inside_geoms_idx]

                filename = f"{S2_download_folder}s2_{y}_" + "_".join(inside_geoms_idx_str) + ".nc"

                geoms_list_dict.append({
                    "idx":inside_geoms_idx.copy(),
                    "aoi":buffered_geom.bounds,
                    "filename":filename})

                inside_geoms_idx.remove(i)
                idx_to_skip = set().union(inside_geoms_idx,idx_to_skip)
            else:
                filename = f"{S2_download_folder}s2_{y}_{i}.nc"
                geoms_list_dict.append({
                    "idx":[i],
                    "aoi":buffered_geom.bounds,
                    "filename":filename})

    # Reorder the dictionary so that we download first the AOIs containing more geometries

    geoms_list_dict = sorted(geoms_list_dict, key=lambda d: len(d["idx"]), reverse=True)

    for gd in geoms_list_dict:

        filename = gd["filename"]
        aoi = gd["aoi"]
        if os.path.exists(filename):
            continue
        else:
            conn = openeo.connect("openeo.cloud").authenticate_oidc()
            # conn = openeo.connect("https://openeo.dataspace.copernicus.eu/openeo/1.2").authenticate_oidc()
            collection = "SENTINEL2_L2A"
            spatial_extent = {"west": aoi[0],
                              "east": aoi[2],
                              "south": aoi[1],
                              "north": aoi[3],
                              "crs":"EPSG:32632"}
            temporal_extent = [f"{y}-04-01",f"{y}-10-31"]
            # s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","SCL"]
            s2_bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]
            s2_cube = conn.load_collection(
                collection,
                spatial_extent = spatial_extent,
                temporal_extent = temporal_extent,
                bands = s2_bands,
                properties = properties
            )
            ## We use the SCL band to create a mask where clouded pixels are set to 1
            ## and other pixels are set to 0
            scl_band = s2_cube.band("SCL")
            s2_cloudmask = ( (scl_band == 8) | (scl_band == 9) | (scl_band == 3) ) * 1.0
            s2_bands.remove("SCL")
            s2_cube_masked = s2_cube.filter_bands(s2_bands).mask(s2_cloudmask)

            # # Create monthly composite
            # composite = s2_cube_masked.aggregate_temporal_period(
            #     period = "month",
            #     reducer = "mean"
            # )
            # # Fill gaps with linear interpolation
            # interpolated = composite.apply_dimension(
            #     dimension = "t",
            #     process = "array_interpolate_linear"
            # )
            # With aggregation + interpolation it takes ~1 minute more per AOI

            from openeo.processes import array_concat, quantiles, sd, mean, median

            res = s2_cube_masked.apply_dimension(
                process=lambda d: array_concat(quantiles(d, [0.1, 0.5, 0.9]), [sd(d), mean(d)]),
                dimension="t",
                target_dimension="bands"
            )

            try:
                start_time = time()
                res.download(filename)
                print("Elapsed time: ", time() - start_time)
            except:
                print(f'{gd["idx"]} failed to download')
                continue

