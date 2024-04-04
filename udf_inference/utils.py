
import urllib.request
import pickle
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import openeo.processes as eop
import openeo
from openeo.processes import text_concat


def preprocess_sentinel2_data(conn: openeo.Connection, spatial_extent: dict, year:int) -> openeo.DataCube:
    """
    Preprocess Sentinel-2 Level-2A data cube.
    - cloud masking
    - aggregate to weekly data
    - interpolate the missing data
    - remove December-January
    - transform from bands, x, y, time to x, y, bands*time

    Args:
        conn: openEO connection object.
        year_param (str): Year for which data is to be loaded.
        spatial_extent_param (dict): Spatial extent for data loading.

    Returns:
        openEO data cube: Preprocessed Sentinel-2 data cube.
    """

    # Load Sentinel-2 Level-2A data cube
    start_day = text_concat([year, "01", "01"], separator="-")
    end_day = text_concat([year, "12", "31"], separator="-")

    s2_cube = conn.load_collection(
        collection_id="SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start_day, end_day],
        bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"],
        properties={"eo:cloud_cover": lambda x: x.lte(65)}
    )

    # Mask data of clouded areas
    s2_cube = s2_cube.process("mask_scl_dilation", data=s2_cube, scl_band_name="SCL").filter_bands(s2_cube.metadata.band_names[:-1])

    # Create weekly composites by taking the mean
    s2_cube = s2_cube.aggregate_temporal_period(
        period="week",
        reducer="mean"
    )

    # Fill gaps in the data using linear interpolation
    s2_cube = s2_cube.apply_dimension(
        dimension="t",
        process="array_interpolate_linear"
    )

    # Filter out January and December data to ensure 43 weeks of data
    feb_day = text_concat([year, "02", "01"], separator="-")
    nov_day = text_concat([year, "11", "30"], separator="-")
    s2_cube = s2_cube.filter_temporal([feb_day, nov_day])

    # Rearrange cube from (time, x, y, bands) to (x, y, time*bands)
    s2_cube = timesteps_as_bands(s2_cube, 43)

    return s2_cube

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


def convert_local_sklearn_to_onnx(model_url: str, input_shape: tuple, output_filename: str) -> None:
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
