import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
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
from pyiceberg.catalog import load_catalog
import numpy as np
import json
from shapely.geometry.point import Point
from typing import Optional
from dreamio_utils import polygon_to_list
from collections import OrderedDict
import warnings

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
    NestedField(field_id=9, name="poi_name", field_type=StringType(), required=False),
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
# Initialize Iceberg Catalog
catalog = load_catalog(
    "nessie",
    **{
        "uri": "http://172.16.100.64:19120/iceberg/main/",
        "s3.endpoint": "http://172.16.100.64:9000",
        "s3.access-key-id": "ZXPByJUzucJ7Y9OkwUhl",
        "s3.secret-access-key": "OSE2YujEDOCvdZR5Gi6pXarB6zja5pvPW6QRHTgl",
    },
)
table_name = "scraping.osm_poi_data"  # Replace with your table name

table = catalog.create_table_if_not_exists(table_name, schema=osm_data_poi_schema)
# catalog.drop_table(table_name)
# raise
# File directory and log file
# cro_location_intelligence/extract_poi_data/data/2024_11_data_7_eleven_poi_dat
file_directory = f"{dir_path}/data/2024_11_data_7_eleven_poi_data/poi_data"  # Replace with your file directory
log_file_path = f"{dir_path}/upload_data_log/poi_data/processed_files.json"

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)


# Load processed files log
def load_processed_files(log_file):
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            return set(json.load(f))
    return set()


# Save processed files log
def save_processed_files(log_file, processed_files):
    with open(log_file, "w") as f:
        json.dump(list(processed_files), f)


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


# Process a single file and append to Iceberg
def process_and_append_file(file_path, iceberg_table, processed_files, log_file):

    try:
        file_name = os.path.basename(file_path)
        table_data = pd.read_pickle(file_path)
        table_data = table_data.reset_index()
        if len(table_data) == 0:
            # file_name
            processed_files.add(file_name)
            save_processed_files(log_file, processed_files)
            return
        processed_table_data = preprocess_table_data(table_data)
        # print("processed_table_data columns ", processed_table_data.columns)
        # Append data to Iceberg
        mapper_ddf = pa.Table.from_pandas(processed_table_data)
        iceberg_table.append(mapper_ddf)
        # print(f"Appended file: {file_path}")

        # Mark file as processed
        processed_files.add(file_name)
        save_processed_files(log_file, processed_files)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


from tqdm.auto import tqdm
from datetime import datetime, timedelta

import sqlite3
from typing import Set, List
import os
from contextlib import contextmanager
import time


class FileTrackingSystem:
    def __init__(self, db_path: str):
        """Initialize the tracking system with SQLite database."""
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create the database and required table if they don't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_files (
                    filename TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time FLOAT
                )
            """
            )
            # Create index for faster lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_filename ON processed_files(filename)"
            )

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def load_processed_files(self) -> Set[str]:
        """Load all processed filenames."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT filename FROM processed_files")
            return set(row[0] for row in cursor.fetchall())

    def get_average_processing_time(self) -> float:
        """Get average processing time per file."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT AVG(processing_time) FROM processed_files")
            avg_time = cursor.fetchone()[0]
            return avg_time if avg_time is not None else 0.0

    def mark_as_processed(self, filenames: List[str], processing_times: List[float]):
        """Mark multiple files as processed with their processing times."""
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO processed_files (filename, processing_time) VALUES (?, ?)",
                zip(filenames, processing_times),
            )
            conn.commit()

    def get_unprocessed_files(self, all_files: List[str]) -> List[str]:
        """Get list of unprocessed files."""
        processed = self.load_processed_files()
        return [f for f in all_files if os.path.basename(f) not in processed]


def process_all_files(
    file_directory: str, iceberg_table, db_path: str, sample_size: int = -1
):
    tracker = FileTrackingSystem(db_path)
    all_files = [os.path.join(file_directory, f) for f in os.listdir(file_directory)]
    unprocessed_files = tracker.get_unprocessed_files(all_files)

    if sample_size != -1:
        unprocessed_files = unprocessed_files[:sample_size]

    total_files = len(unprocessed_files)
    avg_time = tracker.get_average_processing_time()

    print(f"Total files: {len(all_files)}")
    print(f"Unprocessed files: {total_files}")
    if avg_time > 0:
        estimated_total_time = timedelta(seconds=int(avg_time * total_files))
        print(f"Estimated total time: {estimated_total_time}")

    # Process files in batches for better performance
    batch_size = 1000

    # Main progress bar for overall progress
    main_pbar = tqdm(total=total_files, desc="Overall Progress", unit="files")

    for i in range(0, len(unprocessed_files), batch_size):
        batch = unprocessed_files[i : i + batch_size]
        processed_batch = []
        processing_times = []

        # Batch progress bar
        batch_pbar = tqdm(
            batch, desc=f"Batch {i//batch_size + 1}", leave=False, unit="files"
        )

        for file_path in batch_pbar:
            try:
                start_time = time.time()
                file_name = os.path.basename(file_path)

                # Update batch progress bar description with current file
                batch_pbar.set_description(f"Processing {file_name[:30]}...")

                table_data = pd.read_pickle(file_path)
                table_data = table_data.reset_index()

                if len(table_data) > 0:
                    processed_table_data = preprocess_table_data(table_data)
                    mapper_ddf = pa.Table.from_pandas(processed_table_data)
                    iceberg_table.append(mapper_ddf)

                processing_time = time.time() - start_time
                processed_batch.append(file_name)
                processing_times.append(processing_time)

                # Update estimated time remaining
                avg_time = sum(processing_times) / len(processing_times)
                remaining_files = total_files - (i + len(processed_batch))
                est_time_remaining = timedelta(seconds=int(avg_time * remaining_files))
                batch_pbar.set_postfix({"ETA": str(est_time_remaining)})

                # Update main progress bar
                main_pbar.update(1)

            except Exception as e:
                print(f"\nError processing file {file_path}: {e}")

        # Mark batch as processed
        if processed_batch:
            tracker.mark_as_processed(processed_batch, processing_times)

        batch_pbar.close()

    main_pbar.close()

    # Print final statistics
    final_avg_time = tracker.get_average_processing_time()
    print(f"\nProcessing completed!")
    print(f"Average processing time per file: {final_avg_time:.2f} seconds")
    print(f"Total time elapsed: {timedelta(seconds=int(final_avg_time * total_files))}")


# Example usage
if __name__ == "__main__":
    db_path = f"{dir_path}/upload_data_log/poi_data/processed_files.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    sample_size = -1
    process_all_files(file_directory, table, db_path, sample_size)
