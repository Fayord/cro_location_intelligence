import os
import signal
import sys
import traceback
from multiprocessing import Pool, Manager, Value, cpu_count
from ctypes import c_bool
from typing import Optional, List, Dict, Tuple

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely import Point
from tqdm import tqdm

from geo_utils import (
    get_bbox_polygon_from_lat_lon,
    get_features_from_lat_lon,
    tags_dict,
    latlon_crs,
    meter_crs,
)
from geo_save_road_building import select_only_polygon


def init_worker(stop_flag):
    """
    Initialize worker process with interrupt handling.
    This allows each worker to check the global stop flag.
    """

    def handle_interrupt(signum, frame):
        stop_flag.value = True

    signal.signal(signal.SIGINT, handle_interrupt)


def get_building_info_from_lat_lon_within_dist(
    lat: float, lon: float, dist: float, name: str, type3: str
) -> Optional[gpd.GeoDataFrame]:
    """
    Fetch building information within a specified distance of a point.

    Enhanced with more robust error handling and logging.
    """
    try:
        building_data = get_features_from_lat_lon(lat, lon, dist, tags_dict["building"])
    except ox._errors.InsufficientResponseError as e:
        # print(f"Insufficient OSM response for {name}: {e}")
        return None
    except Exception as e:
        # print(f"Error processing {name}: {traceback.format_exc()}")
        # return None
        raise e

    # Proceed with data processing
    building_data = building_data.to_crs(meter_crs)
    building_data = select_only_polygon(building_data)

    # Create coverage polygon
    bbox_polygon = get_bbox_polygon_from_lat_lon(lat, lon, dist)
    polygon_gdf_latlon = gpd.GeoDataFrame(
        {"geometry": bbox_polygon}, index=[0], crs=latlon_crs
    )

    polygon_gdf = polygon_gdf_latlon.to_crs(meter_crs)
    bbox = polygon_gdf.total_bounds
    x_min, y_min, x_max, y_max = bbox
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    middle_point = Point(mid_x, mid_y)

    building_data["middle_point"] = middle_point

    # Vectorized distance calculation
    building_data["distance_to_middle_point"] = building_data.geometry.apply(
        lambda geom: middle_point.distance(geom.centroid)
    )

    # Add metadata
    building_data["latitude"] = lat
    building_data["lontitude"] = lon
    building_data["poi_name"] = name
    building_data["type_lvl03"] = type3

    return building_data


def process_single_poi(args: Tuple[Dict, str, str, int, bool, Value]) -> bool:
    """
    Process a single POI with comprehensive error handling and interrupt support.

    Returns:
        bool: True if processed successfully, False otherwise
    """
    # Unpack arguments
    poi_data, save_folder, error_folder, dist, force, stop_flag = args

    # Check if processing should stop
    if stop_flag.value:
        return False

    lat = float(poi_data["latitude"])
    lon = float(poi_data["longitude"])
    name = poi_data["name"]
    type3 = poi_data["type_lvl03"]
    unique_id = poi_data["unique_id"]

    file_name = f"{unique_id}.pkl"
    file_path = os.path.join(save_folder, file_name)

    # Skip if file exists and not forced
    if os.path.exists(file_path) and not force:
        print(f"File exists: {file_path}")
        return True

    try:
        building_data = get_building_info_from_lat_lon_within_dist(
            lat, lon, dist, name, type3
        )

        if building_data is None:
            building_data = gpd.GeoDataFrame()

        # Save processed data
        building_data.to_pickle(file_path)
        return True

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

        # Save error details
        error_path = os.path.join(error_folder, file_name.replace(".pkl", ".txt"))
        with open(error_path, "w") as f:
            f.write(traceback.format_exc())

        return False


def process_poi_data(poi_data_path: str, sample_size: int = -1) -> List[Dict]:
    """
    Process POI data with efficient deduplication.

    Args:
        poi_data_path (str): Path to the feather file
        sample_size (int): Number of POIs to process. -1 means process all.
    """
    # Read the Feather file
    poi_df = pd.read_feather(poi_data_path)

    # Deduplicate efficiently using pandas
    poi_df_deduped = poi_df.drop_duplicates(subset=["latitude", "longitude", "name"])

    # Optional sampling
    if sample_size != -1:
        poi_df_deduped = poi_df_deduped.head(sample_size)

    # Convert to list of dictionaries for processing
    poi_data_list = poi_df_deduped.apply(
        lambda row: {
            "latitude": row.latitude,
            "longitude": row.longitude,
            "name": row.name,
            "type_lvl03": row.type_lvl03,
            "unique_id": row.unique_id,
        },
        axis=1,
    ).tolist()

    return poi_data_list


def main():
    # Configuration
    IS_FORCE = False
    SAMPLE_DATA = 100  # Change this to control sample size
    SAMPLE_DATA = -1
    DIST = 200  # Search radius in meters

    # Path setup
    dir_path = os.path.dirname(os.path.realpath(__file__))
    poi_data_path = f"{dir_path}/all_pois_relabeled_v4_with_unique_id.feather"
    save_folder = f"{dir_path}/data/2024_11_data_7_eleven_poi_data/poi_data"
    error_folder = f"{dir_path}/data/2024_11_data_7_eleven_poi_data/error_poi_data"

    # Create folders
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)

    # Process POI data
    poi_data_list = process_poi_data(poi_data_path, sample_size=SAMPLE_DATA)
    print(f"Total POIs to process: {len(poi_data_list)}")
    # Create a shared stop flag for interrupt handling
    manager = Manager()
    stop_flag = manager.Value(c_bool, False)

    # Parallel processing with multiprocessing Pool
    with Pool(
        processes=cpu_count(), initializer=init_worker, initargs=(stop_flag,)
    ) as pool:
        # Prepare arguments for each POI processing task
        process_args = [
            (poi_data, save_folder, error_folder, DIST, IS_FORCE, stop_flag)
            for poi_data in poi_data_list
        ]

        # Use imap for lazy evaluation and better interrupt handling
        results = []
        try:
            for result in tqdm(
                pool.imap(process_single_poi, process_args), total=len(process_args)
            ):
                results.append(result)

                # Break if stop flag is set
                if stop_flag.value:
                    print("\n[INTERRUPT] Stopping processing...")
                    break

        except KeyboardInterrupt:
            print("\n[KEYBOARD INTERRUPT] Stopping processing...")
            stop_flag.value = True

    # Print summary
    successful = sum(results)
    print(f"\nProcessing complete. Successful POIs: {successful}/{len(poi_data_list)}")


if __name__ == "__main__":
    main()
