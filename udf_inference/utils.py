
import urllib.request
import pickle
import openeo.processes as eop
import openeo
from openeo.processes import text_concat
import numpy as np

def preprocess_sentinel2_data(conn: openeo.Connection, spatial_extent: dict, temporal_extend:list) -> openeo.DataCube:
    """
    Preprocess Sentinel-2 Level-2A data cube.
    - cloud masking
    - aggregate to weekly data
    - interpolate the missing data
    - median filter time

    Args:
        conn: openEO connection object.
        temporal_extend (list): temporal extend.
        spatial_extent_param (dict): Spatial extent for data loading.

    Returns:
        openEO data cube: Preprocessed Sentinel-2 data cube.
    """

    s2_cube = conn.load_collection(
        collection_id="SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extend,
        bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"],
        properties={"eo:cloud_cover": lambda x: x.lte(65)}
    )

    # Create weekly composites by taking the mean
    s2_cube = s2_cube.process("mask_scl_dilation", data=s2_cube, scl_band_name="SCL").filter_bands(s2_cube.metadata.band_names[:-1])

    # Create weekly composites by taking the mean
    s2_cube = s2_cube.aggregate_temporal_period(
        period = "week",
        reducer = "mean"
    )

    # Fill gaps in the data using linear interpolation
    s2_cube = s2_cube.apply_dimension(
        dimension = "t",
        process = "array_interpolate_linear"
    )

    s2_cube = s2_cube.reduce_dimension(dimension="t",reducer="median")

    return s2_cube

def postprocess_inference_data(input_cube: openeo.DataCube, kernel_size:int) -> openeo.DataCube:
    """
    Apply post-processing operations on inference data represented by a DataCube.

    Parameters:
        input_cube (openeo.DataCube): The input data cube representing the inference data.
        kernel_size (int): The size of the kernel for erosion and dilation operations.

    Returns:
        openeo.DataCube: The post-processed inference data cube.
    """


    kernel = np.ones((kernel_size,kernel_size))
    factor = 1./np.prod(np.shape(kernel))

    eroded_cube = (input_cube.apply_kernel(kernel=kernel,factor=factor) >= 1) * 1.0
    dilated_cube = (eroded_cube.apply_kernel(kernel=kernel,factor=factor) > 0) * 1.0

    return dilated_cube


def convert_sklearn_to_onnx(model_url: str, input_shape: tuple) -> None:
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType
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


def convert_local_sklearn_to_onnx(model_url: str, input_shape: tuple, output_filename: str) -> None:
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType
    """
    Convert a scikit-learn model to ONNX format and save it to the specified output folder.

    Parameters:
        model_url (str): The path from which to load the scikit-learn model.
        output_folder (str): The folder path where the ONNX model will be saved.
        input_shape (tuple): The shape of the input data expected by the model.

    Returns:
        None
    """
    # Load the model from the given path
    with open(model_url,"rb") as f:
        random_forest_model = pickle.load(f)

    # Construct the initial_types argument using FloatTensorType
    input_name = 'input'
    initial_types = [(input_name, FloatTensorType(input_shape))]

    # Convert the model to ONNX
    onnx_model = onnxmltools.convert_sklearn(random_forest_model, initial_types=initial_types)

    # Save the ONNX model to a file
    onnxmltools.utils.save_model(onnx_model, output_filename)
