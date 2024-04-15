# openEO Platform - Photovoltaic Farms mapping

Photovoltaic farms (PV farms) mapping is essential for establishing valid policies regarding natural resources management and clean energy. As evidenced by the recent COP28 summit, where almost 120 global leaders pledged to triple the world’s renewable energy capacity before 2030, it is crucial to make these mapping efforts scalable and reproducible. Recently, there were efforts towards the global mapping of PV farms [1], but these were limited to fixed time periods of the analyzed satellite imagery and not openly reproducible.  Building on this effort, we propose the use of openEO [2] User Defined Processes (UDP) implemented in openEO platform for mapping solar farms using Sentinel-2 imagery, emphasizing the four foundational FAIR data principles: Findability, Accessibility, Interoperability, and Reusability. The UDPs encapsulate the entire workflow including solar farms mapping, starting from data preprocessing and analysis to model training and prediction. The use of openEO UDPs enables easy reuse and parametrization for future PV farms mapping.  

Open-source data is used to construct the training dataset, leveraging OpenStreetMap (OSM) to gather PV farms polygons across different countries. Different filtering techniques are involved in the creation of the training set, in particular land cover and terrain. To ensure model robustness, we leveraged the temporal resolution of Sentinel-2 L2A data and utilized openEO to create a reusable workflow that simplifies the data access in the cloud, allowing the collection of training samples over Europe efficiently. This workflow includes preprocessing steps such as cloud masking, gap filling, outliers filtering as well as feature extraction. Alot of effort is put in the best training samples generation, ensuring an optimal starting point for the subsequent steps. After compiling the training dataset, we conducted a statistical discrimination analysis of different pixel-level models to determine the most effective one. Our goal is to compare time-series machine learning (ML) models like InceptionTime, which uses 3D data as input, with tree-based models like Random Forest (RF), which employs 2D data along with feature engineering. An openEO process graph is then constructed to organize and automate the execution of the inference phase, encapsulating all necessary processes from the preprocessing to the prediction stage. Finally, the process graph is transformed into a reusable UDP that can be reused by others for replicable PV farms mapping, from single farm to country scale. The use of the openEO UDP enables replications of the workflow to map new temporal assessments of PV farms distribution. The UDP process for the PV farms mapping is integrated with the ESA Green Transition Information Factory (GTIF, https://gtif.esa.int/), providing the ability for streamlined and FAIR compliant updates of related energy infrastructure mapping efforts. 

[1] Kruitwagen, L., et al. A global inventory of photovoltaic solar energy generating units. Nature 598, 604–610 (2021). https://doi.org/10.1038/s41586-021-03957-7 

[2] Schramm, M, et al. The openEO API–Harmonising the Use of Earth Observation Cloud Services Using Virtual Data Cube Functionalities. Remote Sens. 2021, 13, 1125. https://doi.org/10.3390/rs13061125 

How to cite: Alasawedah, M., Claus, M., Jacob, A., Griffiths, P., Dries, J., and Lippens, S.: Photovoltaic Farms Mapping using openEO Platform, EGU General Assembly 2024, Vienna, Austria, 14–19 Apr 2024, EGU24-16841, https://doi.org/10.5194/egusphere-egu24-16841, 2024.

![image](https://github.com/masawdah/openEO_photovoltaic/assets/31700619/47f7539d-6442-4f05-81c5-3db5ef022127)

## Video Tutorial


https://github.com/clausmichele/openEO_photovoltaic/assets/31700619/7e312255-2c5b-4894-b913-b536be5ba564



## Usage

You can directly play with the model using openEO Platform or Copernicus Data Space Ecosystem (CDSE) via openEO using this notebook: `./udf_inference/openeo_pv_farms_inference_udf.ipynb`

**Please Note: you need an openEO Platform account to run most of the following scripts. You can get an account [here](https://docs.openeo.cloud/join/free_trial.html).**

## Dataset Generation

### PV Farms geometries

The ground truth geometries for the PV farms are extracted from OpenStreetMap data.

The notebook `./notebooks/photovoltaic.ipynb` explians how PV farms have been extracted from OSM, visualizing the result. (Not ready yet, needs to be refined)

![photovoltaic_osm](https://github.com/masawdah/openEO_photovoltaic/assets/61426508/a466c81c-1f1b-4a76-b530-7a02ab817b65)

### Sentinel-2 Features

We firstly read the geometries and check the associated land cover/land use (LCLU): we want to focus on power plants built on "green" areas.
The script that downloads the LCLU data can be run in this way:

```
python ./scripts/openEO_data_Germany_LANDCOVER.py
```

We use openEO Platform to create the dataset based on Sentinel-2 temporal features.

```
python ./scripts/openEO_data_Germany_S2.py
```

### Random Forest model training

The notebook `./notebooks/rf_training.ipynb` contains the code to train the model over the known PV farms of Germany. (Not ready yet, needs to be refined)

### Model evaluation over Austria

The notebook `./notebooks/rf_validation_austria.ipynb` contains the code to evaluate the model performance over the known PV farms of Austria. (Not ready yet, needs to be refined)

### Additional

The notebook `./notebooks/generate_STAC.ipynb` contains the code to create a STAC Collection based on the resulting openEO netCDF file. This will be used to generate a STAC Collection of the entire dataset soon.

## References

- Global Inventory PV datasets : 
  - https://www.nature.com/articles/s41586-021-03957-7
  - https://zenodo.org/record/5045001
 
- UDF based execution of a CNN model:
  - https://github.com/openEOPlatform/parcel-delineation/blob/main/Parcel%20delineation.ipynb
 
- Dynamically inference based on the process graph or a User Defined Process:
  - https://github.com/Open-EO/openeo-python-client/blob/udp_description/docs/udp.rst#id8


