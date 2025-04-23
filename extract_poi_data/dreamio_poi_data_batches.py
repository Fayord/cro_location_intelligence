import os
import pandas as pd
from pyiceberg.table import Table
from pyiceberg.catalog import load_catalog
from tqdm import tqdm

import json
from shapely.geometry.point import Point
import warnings
from dreamio_utils import polygon_to_list


from pyiceberg.catalog import Catalog
from pyiceberg.schema import Schema, NestedField
from pyiceberg.types import (
    StructType,
    DoubleType,
    ListType,
    IntegerType,
    StringType,
    MapType,
)
import pyarrow as pa


def preprocess_geometry(dataframe, column_name):
    """
    Preprocesses geometry column to convert coordinates to the required format.

    Args:
        dataframe: Input pandas DataFrame
        column_name: Name of the geometry column

    Returns:
        Series containing processed geometry data in format:
        [
            [{"longitude": x1, "latitude": y1}, ...],  # First polygon
            [{"longitude": x2, "latitude": y2}, ...],  # Second polygon (if multipolygon)
            ...
        ]
    """

    def process_polygons(polygons):
        if polygons is None:
            return pd.NA

        try:
            # Convert each polygon's coordinates to dict format
            return [[coord for coord in polygon] for polygon in polygons]
        except Exception as e:
            print(f"Error processing coordinates: {e}")
            return pd.NA

    return dataframe[column_name].apply(process_polygons)


def point_to_list(point: Point) -> list[float]:
    """
    Converts a Shapely Point object to a list of coordinates.

    Args:
        point: A Shapely Point object.

    Returns:
        A list of coordinates, where the single coordinate is a tuple of (x, y) floats.
    """
    return [point.x, point.y]


def preprocess_table_data(df: pd.DataFrame) -> pd.DataFrame:

    # id -> str
    df["id"] = df["id"].astype(str)
    # change name to poi_id
    df = df.rename(columns={"id": "poi_id"})
    # distance_to_middle_point
    df["distance_to_middle_point"] = df["distance_to_middle_point"].astype(float)
    df["middle_point"] = df["middle_point"].apply(
        lambda x: point_to_list(x) if x else None
    )
    # middle_point_lontitude get from middle_point[0]
    df["middle_point_lontitude"] = df["middle_point"].apply(
        lambda x: x[0] if x else None
    )

    # middle_point_latitude get from middle_point[1]
    df["middle_point_latitude"] = df["middle_point"].apply(
        lambda x: x[1] if x else None
    )
    # remove middle_point
    df = df.drop("middle_point", axis=1)
    # poi_name
    df["poi_name"] = df["poi_name"].astype(str)
    # type_lvl03
    df["type_lvl03"] = df["type_lvl03"].astype(str)
    # latitude
    df["latitude"] = df["latitude"].astype(float)
    # lontitude
    df["lontitude"] = df["lontitude"].astype(float)

    df = df.rename(
        columns={
            "latitude": "poi_latitude",
            "lontitude": "poi_lontitude",
        }
    )

    # building_area_dict_string

    all_columns = df.columns.to_list()
    column_except_building_area_dict_string = [
        "poi_id",
        "distance_to_middle_point",
        "middle_point_lontitude",
        "middle_point_latitude",
        "poi_name",
        "type_lvl03",
        "poi_latitude",
        "poi_lontitude",
        "geometry",
    ]
    building_area_dict_string_column = [
        i for i in all_columns if i not in column_except_building_area_dict_string
    ]
    df["building_area_dict_string"] = df[building_area_dict_string_column].apply(
        lambda row: json.dumps(
            {
                col: row[col]
                for col in building_area_dict_string_column
                if not pd.isna(row[col])
            }
        ),
        axis=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        df["geometry"] = df["geometry"].apply(
            lambda x: polygon_to_list(x) if x else None
        )
        df["geometry"] = preprocess_geometry(df, "geometry")
    # remove all building_area_dict_string_column
    df = df.drop(building_area_dict_string_column, axis=1)
    return df


def load_files_in_batches(files, batch_size, iceberg_table):
    """
    Processes and writes files in batches to Iceberg table with progress tracking.

    Args:
        files: List of file paths.
        batch_size: Number of rows per batch.
        iceberg_table: PyIceberg Table object.
    """
    current_batch = []
    current_batch_rows = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(files), desc="Processing Files")

    for file_path in files:
        # Load DataFrame
        df = pd.read_pickle(file_path)

        # Skip files with no rows
        if df.empty:
            pbar.update(1)
            continue
        df = df.reset_index()
        preprocessed_table_data = preprocess_table_data(df)

        # Add to batch
        current_batch.append(preprocessed_table_data)
        current_batch_rows += len(preprocessed_table_data)

        # Write batch to Iceberg when reaching the batch size
        if current_batch_rows >= batch_size:
            write_batch_to_iceberg(current_batch, iceberg_table)
            current_batch = []
            current_batch_rows = 0

        # Update progress bar
        pbar.update(1)

    # Write any remaining rows
    if current_batch:
        write_batch_to_iceberg(current_batch, iceberg_table)

    pbar.close()


def write_batch_to_iceberg(batch, iceberg_table):
    """
    Writes a batch of data to an Iceberg table.
    """
    combined_df = pd.concat(batch, ignore_index=True)
    mapper_ddf = pa.Table.from_pandas(combined_df)
    # raises
    iceberg_table.append(
        mapper_ddf
    )  # Use the appropriate PyIceberg method for appending
    print(f"Wrote batch of {len(combined_df)} rows to Iceberg")
    # raise


# Example Usage
if __name__ == "__main__":

    # Define PyIceberg schema
    osm_data_poi_schema = Schema(
        NestedField(
            field_id=1,
            name="poi_id",
            field_type=StringType(),
            required=False,
        ),
        NestedField(
            field_id=2,
            name="geometry",
            field_type=ListType(
                element_id=3,
                element=ListType(
                    element_id=4,
                    element=ListType(  # Inner list for coordinate pairs [lon, lat]
                        element_id=5,
                        element=DoubleType(),
                        element_required=False,
                    ),
                    element_required=False,
                ),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            field_id=6,
            name="distance_to_middle_point",
            field_type=DoubleType(),
            required=False,
        ),
        NestedField(
            field_id=7,
            name="middle_point_lontitude",
            field_type=DoubleType(),
            required=False,
        ),
        NestedField(
            field_id=8,
            name="middle_point_latitude",
            field_type=DoubleType(),
            required=False,
        ),
        NestedField(
            field_id=9, name="poi_name", field_type=StringType(), required=False
        ),
        NestedField(
            field_id=10, name="type_lvl03", field_type=StringType(), required=False
        ),
        NestedField(
            field_id=11, name="poi_latitude", field_type=DoubleType(), required=False
        ),
        NestedField(
            field_id=12, name="poi_lontitude", field_type=DoubleType(), required=False
        ),
        NestedField(
            field_id=13,
            name="building_area_dict_string",
            field_type=StringType(),
            required=False,
        ),
    )
    dir_path = os.path.dirname(os.path.realpath(__file__))

    dir_folder = f"{dir_path}/data/2024_11_data_7_eleven_poi_data/poi_data"  # Replace with your file directory

    iceberg_catalog = load_catalog(
        "nessie",
        **{
            "uri": "http://172.16.100.64:19120/iceberg/main/",
            "s3.endpoint": "http://172.16.100.64:9000",
            "s3.access-key-id": "ZXPByJUzucJ7Y9OkwUhl",
            "s3.secret-access-key": "OSE2YujEDOCvdZR5Gi6pXarB6zja5pvPW6QRHTgl",
        },
    )
    table_name = "scraping.osm_poi_data"  # Replace with your table name

    iceberg_table = iceberg_catalog.create_table_if_not_exists(
        table_name, schema=osm_data_poi_schema
    )
    # iceberg_catalog.drop_table(table_name)
    # raise
    # Get all pickle files
    files = [
        os.path.join(dir_folder, f)
        for f in os.listdir(dir_folder)
        if f.endswith(".pkl")
    ]
    print("len(files)", len(files))
    # batch_size = 100  # Set batch size, e.g., 1M rows
    batch_size = 5_000_000  # Set batch size, e.g., 1M rows

    load_files_in_batches(files, batch_size, iceberg_table)
