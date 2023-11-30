# openEO Platform - Photovoltaic Farms mapping

The aim of this project is mapping PV farms over Austria, using open source data like OpenStreetMap (OSM) and Sentinel-2.   

The following notebook "photovoltaic.ipynb" explians how PV farms have been extracted from OSM, then visulize the results.
    
![photovoltaic_osm](https://github.com/masawdah/openEO_photovoltaic/assets/61426508/a466c81c-1f1b-4a76-b530-7a02ab817b65)

## Data 
- Global PV Inventory:
  - Training:  25105 polygons from 2017 "trn_polygons.json". 
  - Test: 2082 polygons from 2018 "test_polygons.json".
  - Validation: 500 from 2018 "cv_polygons.json".
 
- photovoltaic_farms.geojson : PV over Austria "not filtered".
  
## References

- Global Inventory PV datasets : 
  - https://www.nature.com/articles/s41586-021-03957-7
  - https://zenodo.org/record/5045001
 
- UDF based execution of a CNN model:
  - https://github.com/openEOPlatform/parcel-delineation/blob/main/Parcel%20delineation.ipynb
 
- Dynamically inference based on the process graph or a User Defined Process:
  - https://github.com/Open-EO/openeo-python-client/blob/udp_description/docs/udp.rst#id8


![photovoltaic_s2_a](https://github.com/masawdah/openEO_photovoltaic/assets/61426508/7a5da5db-fc81-4986-a0ce-f1d235ef426c)

![photovoltaic_s2_b](https://github.com/masawdah/openEO_photovoltaic/assets/61426508/c46cf3b8-bdde-4435-8577-1cae4b630401)
