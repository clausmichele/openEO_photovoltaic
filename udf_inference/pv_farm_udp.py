#%%

"""
UDP to run inference on sentinel 2 data for a preselected area and year. 

The network requires as input 10 bands and 43 weeks

"""

import json
import openeo
from openeo.api.process import Parameter
from utils import  preprocess_sentinel2_data
from openeo.rest.udp import build_process_dict


#establish connection
conn = openeo.connect("https://openeo.dataspace.copernicus.eu/").authenticate_oidc()


#%% Create the process graph

#Changeable parameters include; the spatial extend, the year and the model url.
spatial_extent_param = Parameter(
    name="spatial_extend",
    description="The bounding box to load.",
    schema={
    "type": "object",
    "properties": {
        "west": {"type": "number"},
        "south": {"type": "number"},
        "east": {"type": "number"},
        "north": {"type": "number"},
        "crs": {"type": "string"}
    }
},
)

year_param = Parameter.integer(
    name="year",
    default=2021,
    description="The year for which to generate an annual mean composite.",
)

inference_url_param = Parameter.string(
    name="inference_url",
    default="https://artifactory.vgt.vito.be/artifactory/auxdata-public/photovoltaic/random_forest.onnx",
    description="url to the inference network, must be on VITO artifactory",
)

#%%

s2_cube = preprocess_sentinel2_data(conn,  spatial_extent_param, year_param)

# Supply the model as a URL. The model is stored in artifactory
udf = openeo.UDF.from_file(
    "udf_rf_onnx.py", 
    context={
        "model_url": inference_url_param
    }
)


# Reduce the bands dimesnion to a single prediction using the udf
prediction = s2_cube.reduce_bands(reducer=udf)

#plot the graph
prediction

#%% Stor ethe process graph

#save the graph on the back-end
conn.save_user_defined_process(
    user_defined_process_id="pv_inference",
    process_graph=prediction,
    parameters=[year_param, spatial_extent_param, inference_url_param]
)

#save it locally as a JSON
spec = build_process_dict(
    process_id="pv_inference",
    process_graph=prediction,
    parameters=[year_param, spatial_extent_param, inference_url_param]

)
with open("pv_inference.json", "w") as f:
    json.dump(spec, f, indent=2)

#%% Excecute the process graph
    
# Add the onnx dependencies to the job options. You can reuse this existing dependencies archive
dependencies_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"
job_options = {
    "udf-dependency-archives": [
        f"{dependencies_url} #onnx_deps"
    ],
}

inference_result = conn.datacube_from_process(
    process_id="pv_inference",
    year=2020,
    spatial_extend={"west": 12.17,
                    "east": 12.18,
                    "south": 51.46,
                    "north": 51.47},
    inferene_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/photovoltaic/random_forest.onnx")

inference_result.execute_batch(
    "./photovoltaic_prediction.nc",
    job_options=job_options,
    title="photovoltaic_prediction"
)
