import os

import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional

from geo_utils import (
    get_bbox_polygon_from_lat_lon,
    get_features_from_lat_lon,
    load_data_excel,
    tags_dict,
    latlon_crs,
    meter_crs,
)
import traceback
from geo_save_road_building import select_only_polygon
from shapely import Point
from tqdm import tqdm
import osmnx as ox


def get_building_info_from_lat_lon_within_dist(
    lat: float, lon: float, dist: float, name: str, type3: str
) -> Optional[gpd.GeoDataFrame]:

    try:
        building_data = get_features_from_lat_lon(lat, lon, dist, tags_dict["building"])
    except ox._errors.InsufficientResponseError:
        return None
    except Exception:
        traceback.print_exc()
        raise "Unknown error"
    # buffer road with 1 meter wide
    building_data = building_data.to_crs(meter_crs)
    building_data = select_only_polygon(building_data)
    # create a coverage polygon to create grid

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
    print(middle_point)
    building_data["middle_point"] = middle_point

    # building_data["distance_to_middle_point"] = building_data["middle_point"].distance(
    #     building_data["geometry"].centroid
    # )
    for index, row in building_data.iterrows():
        building_data.loc[index, "distance_to_middle_point"] = row[
            "middle_point"
        ].distance(row["geometry"].centroid)
    # add poi name, lat,lon, typelvl3
    building_data["latitude"] = lat
    building_data["lontitude"] = lon
    building_data["poi_name"] = name
    building_data["type_lvl03"] = type3

    return building_data


def process_poi_data(poi_data_path):
    # Read the Feather file more efficiently
    poi_df = pd.read_feather(poi_data_path)

    # Use set for faster lookup and to avoid duplicates
    check_lat_lon_name_set = set()
    poi_data_list = []

    # Use itertuples() for faster iteration compared to iterrows()
    for row in tqdm(poi_df.itertuples(index=False), total=len(poi_df)):
        # Create unique identifier
        lat_lon_name = (
            row.latitude,
            row.longitude,
            row.name,
        )

        # Skip if already processed
        if lat_lon_name in check_lat_lon_name_set:
            continue

        # Add to set and list
        check_lat_lon_name_set.add(lat_lon_name)
        poi_data = {
            "latitude": row.latitude,
            "longitude": row.longitude,
            "name": row.name,
            "type_lvl03": row.type_lvl03,
            "unique_id": row.unique_id,
        }
        poi_data_list.append(poi_data)

    return poi_data_list


def main():
    IS_FORCE = True
    IS_FORCE = False
    SAMPLE_DATA = -1
    SAMPLE_DATA = 100
    # radius = 200 m
    dist = 200
    # /home/thanatorn/coding/cro_location_intelligence/extract_poi_data/all_pois_relabeled_v4_with_unique_id.feather
    dir_path = os.path.dirname(os.path.realpath(__file__))
    poi_data_path = f"{dir_path}/all_pois_relabeled_v4_with_unique_id.feather"
    save_folder = f"{dir_path}/data/2024_11_data_7_eleven/poi_data"
    error_folder = f"{dir_path}/data/2024_11_data_7_eleven/error_poi_data"
    # create folder if it is not existed
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)

    # poi_df = pd.read_feather(poi_data_path)
    poi_data_list = process_poi_data(poi_data_path)
    if SAMPLE_DATA != -1:
        poi_data_list = poi_data_list[:SAMPLE_DATA]
    print("TOTAL POIS: ", len(poi_data_list))
    for poi_data in tqdm(poi_data_list):
        lat = poi_data["latitude"]
        lon = poi_data["longitude"]
        name = poi_data["name"]
        type3 = poi_data["type_lvl03"]
        unique_id = poi_data["unique_id"]
        lat = float(lat)
        lon = float(lon)
        file_name = f"{unique_id}.pkl"
        file_path = os.path.join(save_folder, file_name)
        # check if file exist and IS_FORCE
        # if file exist and IS_FORCE is False we skip
        if os.path.exists(file_path) and not IS_FORCE:
            print(f"File exists : {file_path}")
            continue
        try:

            building_data = get_building_info_from_lat_lon_within_dist(
                lat, lon, dist, name, type3
            )
            if building_data is None:
                building_data = gpd.GeoDataFrame()
            # save file as pickle
            building_data.to_pickle(file_path)

        except:
            print("error", file_name)
            # traceback
            error_path = os.path.join(error_folder, file_name.replace(".pkl", ".txt"))
            # text file
            with open(error_path, "w") as f:
                f.write(traceback.format_exc())

        # save file
        # except:
        # save on error folder


if __name__ == "__main__":
    main()
