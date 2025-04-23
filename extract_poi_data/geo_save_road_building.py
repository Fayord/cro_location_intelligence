from geo_utils import (
    get_bbox_polygon_from_lat_lon,
    get_features_from_lat_lon,
    load_data_excel,
    tags_dict,
    latlon_crs,
    meter_crs,
)
import osmnx as ox
import pyproj
from collections import Counter
import pandas as pd
from tqdm import tqdm

import json
import time
import os
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Optional

import traceback


def buffer_road_metre(buffer_drive_data, road_size=1):
    # Assuming buffer_drive_data is a pandas DataFrame

    buffer_drive_data["geometry"] = buffer_drive_data.apply(
        lambda x: x.geometry.buffer(road_size / 2), axis=1
    )
    return buffer_drive_data


def grid_points_from_bbox(bbox: List[int], spacing: int = 100, cover_size: int = 1600):
    spacing = int(spacing)
    grid_points = []

    x_min, y_min, x_max, y_max = bbox
    w = cover_size
    h = cover_size
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    x_min = center_x - w / 2
    y_min = center_y - h / 2
    x_max = center_x + w / 2
    y_max = center_y + h / 2
    for x in range(int(x_min) + spacing // 2, int(x_max), spacing):
        for y in range(int(y_min) + spacing // 2, int(y_max), spacing):
            point = Point(x, y)
            grid_points.append(point)
    return grid_points


def select_only_polygon(building_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # print("before")
    # print(Counter(building_data.geometry.type))
    # Polygon or MultiPolygon is allowed
    building_data = building_data[
        (building_data["geometry"].type == "Polygon")
        | (building_data["geometry"].type == "MultiPolygon")
    ]
    # print("after")
    # print(Counter(building_data.geometry.type))
    return building_data


def get_building_info_from_lat_lon(
    lat, lon, dist, grid_size=100
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
    # convert to meter
    polygon_gdf = polygon_gdf_latlon.to_crs(meter_crs)
    bbox = polygon_gdf.total_bounds
    # create grid points
    grid_points = grid_points_from_bbox(bbox, spacing=grid_size, cover_size=dist * 2)
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=meter_crs)

    # buffer grid points to make small square over the grid points
    buffered_grid_gdf = grid_gdf
    buffered_grid_gdf["geometry"] = buffered_grid_gdf["geometry"].buffer(
        grid_size / 2, cap_style=3
    )
    buffered_grid_gdf["grid_id"] = buffered_grid_gdf.index

    # get intersection between grid and road to extract each kinds of road area
    # print(building_data.shape)
    # print(building_data.head())
    # print(buffered_grid_gdf.shape)
    # print(buffered_grid_gdf.head())
    intersection_gdf = gpd.overlay(
        buffered_grid_gdf,
        building_data,
        how="intersection",
        # on=["geometry", "geometry"],
    )
    # print(intersection_gdf.__dict__)
    # print(intersection_gdf.building.unique())
    # raise
    buffered_grid_gdf["building_area"] = 0.0
    for key in intersection_gdf.building.unique():
        buffered_grid_gdf["building_area_" + key] = 0.0
    for index, row in intersection_gdf.iterrows():
        grid_id = row["grid_id"]
        building_type = row["building"]

        # print(type(building_type))

        area = row["geometry"].area
        # print(type(area))
        # print("building_type", building_type)
        buffered_grid_gdf.loc[grid_id, "building_area_" + building_type] += area
    for index in buffered_grid_gdf.index:
        grid_id = index
        grid_number_one_side = dist * 2 // grid_size
        grid_lat_id = int(grid_id % grid_number_one_side)
        grid_lon_id = int(grid_id // grid_number_one_side)
        buffered_grid_gdf.loc[grid_id, "grid_lat_id"] = grid_lat_id
        buffered_grid_gdf.loc[grid_id, "grid_lon_id"] = grid_lon_id
    # sum road length
    for key in intersection_gdf.building.unique():
        buffered_grid_gdf["building_area"] += buffered_grid_gdf["building_area_" + key]
    # drop grid_id
    buffered_grid_gdf = buffered_grid_gdf.drop(columns=["grid_id"])
    return buffered_grid_gdf


def get_road_info_from_lat_lon(lat, lon, dist, grid_size=100) -> gpd.GeoDataFrame:
    # get road info from lat lon
    drive_data = get_features_from_lat_lon(lat, lon, dist, tags_dict["highway"])

    # buffer road with 1 meter wide
    drive_data = drive_data.to_crs(meter_crs)
    buffer_drive_data = buffer_road_metre(drive_data, road_size=1)
    # create a coverage polygon to create grid
    bbox_polygon = get_bbox_polygon_from_lat_lon(lat, lon, dist)
    polygon_gdf_latlon = gpd.GeoDataFrame(
        {"geometry": bbox_polygon}, index=[0], crs=latlon_crs
    )
    # convert to meter
    polygon_gdf = polygon_gdf_latlon.to_crs(meter_crs)
    bbox = polygon_gdf.total_bounds
    # create grid points
    grid_points = grid_points_from_bbox(bbox, spacing=grid_size, cover_size=dist * 2)
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=meter_crs)

    # buffer grid points to make small square over the grid points
    buffered_grid_gdf = grid_gdf
    buffered_grid_gdf["geometry"] = buffered_grid_gdf["geometry"].buffer(
        grid_size / 2, cap_style=3
    )
    buffered_grid_gdf["grid_id"] = buffered_grid_gdf.index

    # get intersection between grid and road to extract each kinds of road area
    intersection_gdf = gpd.overlay(
        buffered_grid_gdf, buffer_drive_data, how="intersection"
    )

    buffered_grid_gdf["road_length"] = 0.0
    for key in intersection_gdf.highway.unique():
        buffered_grid_gdf["road_length_" + key] = 0.0
    for index, row in intersection_gdf.iterrows():
        grid_id = row["grid_id"]
        road_type = row["highway"]
        area = row["geometry"].area
        # print("road_type", road_type)
        buffered_grid_gdf.loc[grid_id, "road_length_" + road_type] += area
    for index in buffered_grid_gdf.index:
        grid_id = index
        grid_number_one_side = dist * 2 // grid_size
        grid_lat_id = int(grid_id % grid_number_one_side)
        grid_lon_id = int(grid_id // grid_number_one_side)
        buffered_grid_gdf.loc[grid_id, "grid_lat_id"] = grid_lat_id
        buffered_grid_gdf.loc[grid_id, "grid_lon_id"] = grid_lon_id
    # sum road length
    for key in intersection_gdf.highway.unique():
        buffered_grid_gdf["road_length"] += buffered_grid_gdf["road_length_" + key]
    # drop grid_id
    buffered_grid_gdf = buffered_grid_gdf.drop(columns=["grid_id"])
    return buffered_grid_gdf


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    IS_FORCE = True
    IS_FORCE = False
    project = "chester"
    project = "7-eleven"
    project = "2024_11_7-eleven"
    grid_size = 100
    dist = 1600 // 2
    if project == "chester":
        data_path = f"{dir_path}/chester_branch_post_process.xlsx"
        save_folder = f"{dir_path}/data_chester/road_info"
        error_folder = f"{dir_path}/data_chester/error"
    elif project == "7-eleven":
        data_path = f"{dir_path}/7-11 Location for Ford.xlsx"
        data_path = f"{dir_path}/ร้านใหม่(3).csv"
        save_folder = f"{dir_path}/data_7_eleven/road_info"
        error_folder = f"{dir_path}/data_7_eleven/error"
    elif project == "2024_11_7-eleven":
        data_path = f"{dir_path}/2024_11_store_profile.csv"
        save_folder = f"{dir_path}/data/2024_11_data_7_eleven/road_info"
        error_folder = f"{dir_path}/data/2024_11_data_7_eleven/error"
        grid_size = 500
        dist = 5000 // 2
    else:
        raise ValueError("project must be chester or 7-eleven")
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)
    try:
        df = load_data_excel(data_path)
    except:
        df = pd.read_csv(data_path)
    number_tiles_aspected = dist * dist * 4 / grid_size / grid_size
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            lat = row["latitude"]
            lon = row["longitude"]
            store_id = row["store_id"]
            store_id = int(store_id)
            save_path = os.path.join(save_folder, f"{store_id}.pkl")
            if not IS_FORCE and os.path.exists(save_path):
                print(f"skip {store_id}")
                continue

            road_info_gdf = get_road_info_from_lat_lon(
                lat, lon, dist, grid_size=grid_size
            )
            # rename geometry column to road_geometry
            building_info_gdf = get_building_info_from_lat_lon(
                lat, lon, dist, grid_size=grid_size
            )
            if building_info_gdf is None:
                merge_info_gdf = road_info_gdf
            else:
                road_info_gdf = road_info_gdf.rename(
                    columns={"geometry": "road_geometry"}
                )
                building_info_gdf = building_info_gdf.rename(
                    columns={"geometry": "building_geometry"}
                )
                merge_info_gdf = road_info_gdf.merge(
                    building_info_gdf, on=["grid_lat_id", "grid_lon_id"], how="outer"
                )
            # add store_id column
            merge_info_gdf["store_id"] = store_id
            assert (
                merge_info_gdf.shape[0] == number_tiles_aspected
            ), f"{store_id} {merge_info_gdf.shape[0]} != {number_tiles_aspected}"
            # save as pickle
            merge_info_gdf.to_pickle(save_path, protocol=4)
        except:
            # save textfile to error folder
            file_path = os.path.join(error_folder, f"{store_id}.txt")
            # save textfile
            with open(file_path, "w") as f:
                f.write(f"{store_id}")
            print(f"Error: {store_id}")


if __name__ == "__main__":
    main()
