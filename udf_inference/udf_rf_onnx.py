import sys
from typing import Dict
import xarray as xr
from openeo.udf import inspect
import requests
import numpy as np
import functools

# Import the onnxruntime package from the onnx_deps directory
sys.path.insert(0, "onnx_deps")
import onnxruntime as ort


@functools.lru_cache(maxsize=6)
def _load_ort_session(model_url: str):
    """
    Load the models and make the prediction functions.
    The lru_cache avoids loading the model multiple times on the same worker.
    """
    inspect(message=f"Loading random forrest as ONNX runtime session ...")
    response = requests.get(model_url)
    model = response.content
    inspect(message=f"Model loaded from {model_url}", level="debug")
    return ort.InferenceSession(model)

def _apply_ml(input_data : np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """
    Apply the model to a tensor containing features.
    """
    return session.run(None, {'input': input_data})[0]
    

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Apply the model to the datacube.
    """
    # Load the model
    session = _load_ort_session(context.get("model_url", None))

    # Prepare the input
    cube = cube.values.astype(np.float32)
    input_data = np.nan_to_num(cube, nan=-999999)
    
    #reshape to desired format
    input_data = input_data.reshape((input_data.shape[0], -1)).T


    # Make the prediction
    output = _apply_ml(input_data, session)
    # Prepare the output
    output = output.reshape(cube.shape[1:])
    return xr.DataArray(output)

