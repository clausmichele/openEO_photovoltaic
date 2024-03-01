# openEO Platform - Photovoltaic Farms mapping

The aim of this project is mapping PV farms over Austria, using open source data like OpenStreetMap (OSM) and Sentinel-2.   

**Please Note: you need an openEO Platform account to run most of the following scripts. You can get an account [here](https://docs.openeo.cloud/join/free_trial.html).**
## Dataset Generation

The training set will get samples from Germany only for the moment. The area could be expanded in a second stage to include samples from more countries.

### PV Farms geometries

The following notebook "photovoltaic.ipynb" explians how PV farms have been extracted from OSM, then visulize the results.
    
![photovoltaic_osm](https://github.com/masawdah/openEO_photovoltaic/assets/61426508/a466c81c-1f1b-4a76-b530-7a02ab817b65)

**Selected Geometries** 
  - Germany:  3687 PV farms. 
  - Austria:  43 PV farms.

### Sentinel-2 Features

We firstly read the geometries and check the associated land cover/land use (LCLU): we want to focus on power plants built on "green" areas.
The script that downloads the LCLU data can be run in this way:

```
python ./scripts/openEO_data_Germany_LANDCOVER.py
```

We use openEO Platform to create the Dataset based on Sentinel-2 temporal features such as mean, median, standard deviation, 10% and 90% quantiles.


```
python ./scripts/openEO_data_Germany_S2.py
```
  
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
