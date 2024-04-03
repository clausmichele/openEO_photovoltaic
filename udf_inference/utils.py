
import urllib.request
import pickle
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import openeo.processes as eop
import openeo

def timesteps_as_bands(cube: openeo.DataCube, n_times:int) -> openeo.DataCube:

    """
    Transforms the time dimension of a multi-temporal data cube into a multi-band data cube,
    where each band represents a time step.

    Parameters:
    - cube (openeo.DataCube): The input multi-temporal data cube.
    - n_times (int): The number of time steps to consider.

    Returns:
    - openeo.DataCube: A new data cube with time steps transformed into bands.

    """

    band_names = [band + "_t" + str(i+1) for band in cube.metadata.band_names for i in range(n_times)]
    result =  cube.apply_dimension(
        dimension='t', 
        target_dimension='bands', 
        process=lambda d: eop.array_create(data=d)
    )
    return result.rename_labels('bands', band_names)


def convert_sklearn_to_onnx(model_url: str, input_shape: tuple) -> None:
    """
    Convert a scikit-learn model to ONNX format and save it to the specified output folder.

    Parameters:
        model_url (str): The URL from which to load the scikit-learn model.
        output_folder (str): The folder path where the ONNX model will be saved.
        input_shape (tuple): The shape of the input data expected by the model.

    Returns:
        None
    """
    # Load the model from the given URL
    with urllib.request.urlopen(model_url) as model_file:
        random_forest_model = pickle.load(model_file)

    # Construct the initial_types argument using FloatTensorType
    input_name = 'input'
    initial_types = [(input_name, FloatTensorType(input_shape))]

    # Convert the model to ONNX
    onnx_model = onnxmltools.convert_sklearn(random_forest_model, initial_types=initial_types)

    # Save the ONNX model to a file
    onnxmltools.utils.save_model(onnx_model, "random_forest.onnx")

