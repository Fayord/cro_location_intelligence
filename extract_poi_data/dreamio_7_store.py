import os
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField,
    StringType,
    DoubleType,
    ListType,
    StructType,
    LongType,
    IntegerType,
)
from shapely.geometry import Polygon
import pandas as pd

import pyarrow as pa
import json
import numpy as np


def polygon_to_list(polygon):
    """
    Converts a Shapely Polygon object to a list of coordinates.

    Args:
        polygon: A Shapely Polygon object.

    Returns:
        A list of coordinates, where each coordinate is a tuple of (x, y) floats.
    """
    coords = list(polygon.exterior.coords)
    # Remove the duplicate first and last coordinate
    return coords[:-1]


def preprocess_geometry(dataframe, column_name):
    def flatten(row):
        try:
            return [{"longitude": lon, "latitude": lat} for lon, lat in row]
        except:
            return pd.NA

    return dataframe[column_name].apply(flatten)


def preprocess_road_building_data(df: pd.DataFrame) -> pd.DataFrame:
    # column grid_lat_id, grid_lon_id convert to int
    df["grid_lat_id"] = df["grid_lat_id"].astype(np.int16)
    df["grid_lon_id"] = df["grid_lon_id"].astype(np.int16)
    # store_id to string
    df["store_id"] = df["store_id"].astype(str)

    # get column name that have road_building in it name
    column_list = df.columns.to_list()
    road_length_column_list = [i for i in column_list if "road_length" in i]
    print(len(road_length_column_list))
    building_area_column_list = [i for i in column_list if "building_area" in i]
    print(len(building_area_column_list))
    # Convert road_length columns to a dictionary and then to a JSON string
    df["road_length_dict_string"] = df[road_length_column_list].apply(
        lambda row: json.dumps(
            {col: row[col] for col in road_length_column_list if not pd.isna(row[col])}
        ),
        axis=1,
    )

    # Convert building_area columns to a dictionary and then to a JSON string
    df["building_area_dict_string"] = df[building_area_column_list].apply(
        lambda row: json.dumps(
            {
                col: row[col]
                for col in building_area_column_list
                if not pd.isna(row[col])
            }
        ),
        axis=1,
    )

    # apply polygon_to_list to column road_geometry
    df["road_geometry"] = df["road_geometry"].apply(
        lambda x: polygon_to_list(x) if x else None
    )
    df["building_geometry"] = df["building_geometry"].apply(
        lambda x: polygon_to_list(x) if x else None
    )

    # remove road_length_column_list and building_area_column_list
    df = df.drop(road_length_column_list, axis=1)
    df = df.drop(building_area_column_list, axis=1)

    return df


def main():
    catalog = load_catalog(
        "nessie",
        **{
            "uri": "http://172.16.100.64:19120/iceberg/main/",
            "s3.endpoint": "http://172.16.100.64:9000",
            "s3.access-key-id": "ZXPByJUzucJ7Y9OkwUhl",
            "s3.secret-access-key": "OSE2YujEDOCvdZR5Gi6pXarB6zja5pvPW6QRHTgl",
        },
    )
    table_7_name = f"scraping.osm_7_store_data"
    osm_data_7_schema = Schema(
        NestedField(
            field_id=1,
            name="road_geometry",
            field_type=ListType(
                element_id=2,
                element=StructType(
                    NestedField(
                        field_id=3,
                        name="longitude",
                        field_type=DoubleType(),
                        required=False,
                    ),
                    NestedField(
                        field_id=4,
                        name="latitude",
                        field_type=DoubleType(),
                        required=False,
                    ),
                ),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            field_id=5,
            name="building_geometry",
            field_type=ListType(
                element_id=6,
                element=StructType(
                    NestedField(
                        field_id=7,
                        name="longitude",
                        field_type=DoubleType(),
                        required=False,
                    ),
                    NestedField(
                        field_id=8,
                        name="latitude",
                        field_type=DoubleType(),
                        required=False,
                    ),
                ),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            field_id=6,
            name="store_id",
            field_type=StringType(),
            required=False,
        ),
        NestedField(
            field_id=7,
            name="grid_lat_id",
            field_type=IntegerType(),
            required=False,
        ),
        NestedField(
            field_id=8,
            name="grid_lon_id",
            field_type=IntegerType(),
            required=False,
        ),
        NestedField(
            field_id=9,
            name="building_area_dict_string",
            field_type=StringType(),
            required=False,
        ),
        NestedField(
            field_id=10,
            name="road_length_dict_string",
            field_type=StringType(),
            required=False,
        ),
    )

    merge_road_building_data_path = "/home/thanatorn/coding/cro_location_intelligence/extract_poi_data/merge_road_building_data.pkl"
    merge_road_building_data = pd.read_pickle(merge_road_building_data_path)
    sample_merge_road_building_data = merge_road_building_data
    processed_road_building_data = preprocess_road_building_data(
        sample_merge_road_building_data
    )

    processed_road_building_data["road_geometry"] = preprocess_geometry(
        processed_road_building_data, "road_geometry"
    )
    processed_road_building_data["building_geometry"] = preprocess_geometry(
        processed_road_building_data, "building_geometry"
    )
    mapper_ddf = pa.Table.from_pandas(processed_road_building_data)
    # Initialize the Iceberg catalog
    # catalog = Catalog.load("your-catalog-name")  # Replace with your catalog name or URI

    # Create the table if it does not exist

    table = catalog.create_table_if_not_exists(table_7_name, schema=osm_data_7_schema)

    # Append data to the Iceberg table
    table.append(mapper_ddf)
    print("complete")


if __name__ == "__main__":
    main()
    # # Example usage:
    # polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    # coordinates = polygon_to_list(polygon)
    # print(coordinates)  # Output: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # converted_polygon = Polygon(coordinates)
    # print(converted_polygon == polygon)
