import sys
from typing import Dict
import xarray as xr
from openeo.udf import inspect
import requests
import numpy as np
import functools
import urllib
import shutil
import numpy as np
import os
from pathlib import Path

SessionType = type(None)

@functools.lru_cache(maxsize=6)
def extract_ort(dependency_dir: str, dependency_name: str):

    """
    Extracts the ONNX Runtime (ORT) model from the specified directory.

    Parameters:
        dependency_dir (str): The directory where the ONNX dependency is located.
        dependency_name (str): The tag or identifier of the ONNX model.

    Returns:
        onnxruntime.InferenceSession: An InferenceSession object representing the extracted model.

    Note:
        This function retrieves the ONNX model specified by `modeltag` from the `modeldir`.
        It then unpacks the model file and imports ONNX Runtime to enable inference.
        The extracted model is returned as an InferenceSession object.
    """

    modelfile = str(dependency_dir) + '/' + dependency_name
    extract_dir = Path.cwd() / 'dependencies'
    extract_dir.mkdir(exist_ok=True, parents=True)

    modelfile, _ = urllib.request.urlretrieve(
        modelfile, filename=extract_dir / Path(modelfile).name)

    shutil.unpack_archive(modelfile,
                          extract_dir=extract_dir)

    sys.path.append(str(extract_dir))

    # Import onnxruntime here as well
    import onnxruntime as ort
    return ort 
    
      

def _load_ort_session(model_url: str) -> SessionType:
    """
    Load the ONNX runtime session for the given model URL, ensuring required dependencies are available.

    Parameters:
    - model_url (str): The URL of the model to load.
    - dependency_url (str): The URL of the dependency required by the model.
    - extract_dir (str): The directory to extract the dependency into.

    Returns:
    - ort.InferenceSession: The ONNX runtime session loaded with the model.
    """
    # Download and extract the required dependency
    dependency_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo"
    dependency_name = "onnx_dependencies_1.16.3.zip"

    inspect(message=f"Including Dependency ONNX Inference")
    ort = extract_ort(dependency_url, dependency_name)

    # Adapt Sessiontype
    global SessionType
    SessionType = ort.InferenceSession
    inspect(message=f"Dependencies loaded from {dependency_url}", level="debug")

    # Load the model
    inspect(message=f"Loading random forrest as ONNX runtime session ...")
    response = requests.get(model_url)
    model = response.content
    inspect(message=f"Model loaded from {model_url}", level="debug")

    # Load the dependency into an InferenceSession
    session = ort.InferenceSession(model)

    # Return the ONNX runtime session loaded with the model
    return session



def _apply_ml(input_data: np.ndarray, session: SessionType) -> np.ndarray:
    """
    Apply the model to a tensor containing features.
    """
    if session is None:
        # Handle the case when session is not yet assigned
        raise ValueError("Inference session is not initialized yet.")
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

