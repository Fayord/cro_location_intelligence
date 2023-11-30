import osmnx as ox
import pandas as pd
import geopandas as gpd
from typing import Tuple
import utm
import pyproj
from PIL import Image
import os
from osmnx.plot import _save_and_show
from shapely.geometry import Point
import webcolors
import matplotlib.figure as figure

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

import osmnx as ox

import pyproj

latlon_crs = pyproj.CRS("EPSG:4326")
meter_crs = pyproj.CRS("EPSG:3857")


def create_circle_polygon(center_lat: float, center_lon: float, radius: float):
    # Create a Point geometry for the circle center
    center_point = Point(center_lon, center_lat)
    latlon_crs = pyproj.CRS("EPSG:4326")
    gdf = gpd.GeoDataFrame(geometry=[center_point])
    gdf = gdf.set_crs(latlon_crs)
    gdf["geometry"] = gdf.geometry.buffer(radius)

    return gdf


def get_rgb_from_color_name(color_name: str) -> tuple:
    if color_name == "green":
        color_name = "lime"
    try:
        # Get RGB values for the color name
        rgb_values = webcolors.name_to_rgb(color_name)
        # print(f"RGB values for {color_name}: {rgb_values}")
    except ValueError:
        raise f"Unable to find RGB values for {color_name}"
    rgb_tuple = tuple([x for x in rgb_values])
    return rgb_tuple


dir_path = os.path.dirname(os.path.realpath(__file__))
highway_config = {
    "motorway": {"size": 6, "color": "red"},
    "trunk": {"size": 6, "color": "red"},
    "primary": {"size": 6, "color": "red"},
    "secondary": {"size": 5, "color": "purple"},
    "tertiary": {"size": 4, "color": "blue"},
    "unclassified": {"size": 3, "color": "blue"},
    "residential": {"size": 1, "color": "blue"},
    "other": {"size": 1, "color": "blue"},
}
building_config = {
    "=1": {"color_density_level": 1, "color": "green"},
    "=2": {"color_density_level": 2, "color": "green"},
    "=3": {"color_density_level": 3, "color": "green"},
    "=4": {"color_density_level": 4, "color": "green"},
    ">4": {"color_density_level": 5, "color": "green"},
}


def calculate_color_density(building_config: dict) -> dict:
    color_density = {}
    for key, value in building_config.items():
        color_density[key] = value["color_density_level"]
    min_color_density = min(color_density.values())
    max_color_density = max(color_density.values())
    # rescale it to 0-255
    for key, value in color_density.items():
        color_density[key] = int(
            (value - min_color_density + 1)
            / (max_color_density - min_color_density + 1)
            * 255
        )
    # update building_config
    for key, value in building_config.items():
        value["color_density"] = color_density[key]
    return building_config


building_config = calculate_color_density(building_config)
tags_dict = {
    "building": {"building": True},
    "highway": {"highway": True},
    "amenity": {"amenity": True},
    "landuse": {"landuse": True},
}


# add road_size according to highway_config
def get_road_size(highway_value, default_size):
    return highway_config.get(highway_value, {"size": default_size})["size"]


def buffer_road(buffer_drive_data):
    # Assuming buffer_drive_data is a pandas DataFrame
    if buffer_drive_data is None:
        return None
    buffer_drive_data["road_size"] = buffer_drive_data["highway"].apply(
        lambda x: get_road_size(x, default_size=highway_config["other"]["size"])
    )
    buffer_drive_data["geometry"] = buffer_drive_data.apply(
        lambda x: x.geometry.buffer(0.00001 * x.road_size), axis=1
    )
    return buffer_drive_data


def draw_building(processed_building_data, ax, edge_linewidth, plot_bbox, dpi):
    fig = None
    edge_color = get_rgb_from_color_name("green")
    edge_color = [int(rgb) for rgb in edge_color]
    edge_color = "#{:02x}{:02x}{:02x}".format(*edge_color)
    for floor_condition, config in building_config.items():
        expression = floor_condition[0]
        expression_value = int(floor_condition[1:])

        rgb_tuple = get_rgb_from_color_name(config["color"])
        rgb_tuple = [int(rgb / 255 * config["color_density"]) for rgb in rgb_tuple]
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)
        if expression == "=":
            selected_data = processed_building_data[
                processed_building_data["building:levels"] == expression_value
            ]
        elif expression == ">":
            selected_data = processed_building_data[
                processed_building_data["building:levels"] > expression_value
            ]
        elif expression == "<":
            selected_data = processed_building_data[
                processed_building_data["building:levels"] < expression_value
            ]
        else:
            raise f"Expression {expression} is not supported"
        if selected_data.shape[0] == 0:
            continue
        # print("selected_data", selected_data.shape)
        fig, ax = ox.plot_footprints(
            selected_data,
            ax=ax,
            alpha=0.4,
            edge_linewidth=edge_linewidth,
            edge_color=edge_color,
            show=False,
            color=hex_color,
            bbox=plot_bbox,
            dpi=dpi,
        )
    return fig, ax


def draw_drive(buffer_drive_data, ax, edge_linewidth, plot_bbox, dpi):
    fig = None
    road_type_list = list(highway_config.keys())
    road_type_list.remove("other")
    road_type_list = ["other"] + road_type_list
    for road_type in road_type_list:
        config = highway_config[road_type]
        rgb_tuple = get_rgb_from_color_name(config["color"])
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)
        if road_type == "other":
            selected_data = buffer_drive_data
        else:
            selected_data = buffer_drive_data[buffer_drive_data["highway"] == road_type]
        if selected_data.shape[0] == 0:
            continue
        fig, ax = ox.plot_footprints(
            selected_data,
            ax=ax,
            alpha=1,
            # edge_linewidth=edge_linewidth,
            # edge_color="white",
            show=False,
            color=hex_color,
            bbox=plot_bbox,
            dpi=dpi,
        )
    return fig, ax


def draw_circle(circle_polygon, ax, edge_linewidth, plot_bbox, dpi):
    fig = None

    fig, ax = ox.plot_footprints(
        circle_polygon,
        ax=ax,
        alpha=1,
        show=False,
        color="white",
        bbox=plot_bbox,
        dpi=dpi,
    )
    return fig, ax


def create_full_image_from_lat_lon(
    lat: float, lon: float, dist: float, edge_linewidth, dpi
) -> figure.Figure:
    polygon = get_bbox_polygon_from_lat_lon(lat, lon, dist)

    building_data = get_features_from_lat_lon(lat, lon, dist, tags_dict["building"])
    polygon_gdf = gpd.GeoDataFrame({"geometry": polygon}, index=[0], crs=latlon_crs)
    drive_data = get_features_from_lat_lon(lat, lon, dist, tags_dict["highway"])

    buffer_drive_data = buffer_road(drive_data)
    plot_bbox = get_plot_bbox_from_polygon_gdf(polygon_gdf)

    radius_meters = 0.0001  # Adjust the radius as needed
    circle_polygon = create_circle_polygon(lat, lon, radius_meters)

    ax = None
    processed_building_data = preprocess_features(
        building_data,
        tags=tags_dict["building"],
    )
    # remove point
    processed_building_data = processed_building_data[
        processed_building_data["geometry"].type != "Point"
    ]
    fig, ax = draw_building(
        processed_building_data, ax, edge_linewidth, plot_bbox=plot_bbox, dpi=dpi
    )
    fig, ax = draw_drive(
        buffer_drive_data, ax, edge_linewidth, plot_bbox=plot_bbox, dpi=dpi
    )
    # fig, ax = draw_circle(
    #     circle_polygon, ax, edge_linewidth, plot_bbox=plot_bbox, dpi=dpi
    # )
    return fig, ax


def save_full_image_from_lat_lon(lat, lon, store_id, dist, dpi, edge_linewidth):
    save_filename = f"{store_id}.png"
    save_path = os.path.join(dir_path, f"data/full_image/{save_filename}")
    if os.path.exists(save_path):
        print(f"File exists : {save_path}")
        return
    cover_dist = get_cover_radius_size(dist)
    fig, ax = create_full_image_from_lat_lon(lat, lon, cover_dist, edge_linewidth, dpi)
    # fig.savefig(save_path, dpi=dpi)
    _save_and_show(
        fig, ax, save=True, show=False, close=True, filepath=save_path, dpi=dpi
    )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # int_columns = ["store_id", "mockup_sale"]
    # int_columns = ["store_id"]
    float_columns = ["latitude", "longitude"]
    # df[int_columns] = df[int_columns].astype(int)
    df[float_columns] = df[float_columns].astype(float)
    return df


def load_data_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = preprocess_data(df)
    return df


def apply_circle_mask(image: Image.Image):
    image = image.copy()
    width, height = image.size
    r = min(width, height) / 2
    x = width / 2
    y = height / 2
    for i in range(width):
        for j in range(height):
            if (i - x) ** 2 + (j - y) ** 2 > r**2:
                image.putpixel((i, j), (0, 0, 0))
    return image


def crop_middle_image_from_side_size(
    image: Image.Image,
    original_side_size: float,
    target_side_size: float,
) -> Image.Image:
    # get image size
    width, height = image.size
    crop_width = int(width * target_side_size / original_side_size)
    crop_height = int(height * target_side_size / original_side_size)
    # crop image
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2
    image = image.crop((left, top, right, bottom))
    return image


def lat_lon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """
    Convert latitude and longitude coordinates to UTM coordinates in meters.

    Parameters:
    - lat (float): Latitude in decimal degrees.
    - lon (float): Longitude in decimal degrees.

    Returns:
    - tuple: UTM coordinates in meters (easting, northing, zone number, zone letter).
    """
    utm_coords = utm.from_latlon(lat, lon)
    return utm_coords[0], utm_coords[1], utm_coords[2], utm_coords[3]


def lat_lon_to_meter(lat: float, lon: float) -> Tuple[float, float]:
    utm_coords = lat_lon_to_utm(lat, lon)
    return utm_coords[0], utm_coords[1]


# function to calculate cover radius to cover all angle of small radius
def get_cover_radius_size(size: float, padding: float = 1.5) -> float:
    # in calculation least cover radius, we need is diagonal of small radius = size*sqrt(2)
    # sqrt(2) is about 1.4 but for make sure we use 1.5
    size = size * padding
    return size


from shapely.validation import make_valid, explain_validity


def get_bbox_polygon_from_lat_lon(
    lat: float, lon: float, distance: float
) -> gpd.GeoDataFrame:
    # distance is radius in meter
    bbox = ox.utils_geo.bbox_from_point((lat, lon), dist=distance)

    # create a new buffer polygon from this bbox geometry
    polygon = ox.utils_geo.bbox_to_poly(*bbox)

    return polygon


import traceback


def get_features_from_lat_lon(
    lat: float,
    lon: float,
    distance: float = 100,
    tags: dict = {"building": True},
) -> gpd.GeoDataFrame:
    cover_dist = get_cover_radius_size(distance)
    # create a new buffer polygon from this bbox geometry
    polygon = get_bbox_polygon_from_lat_lon(lat, lon, cover_dist)

    return ox.features_from_polygon(polygon, tags)


def get_plot_bbox_from_polygon_gdf(
    polygon: gpd.GeoDataFrame,
) -> Tuple[float, float, float, float]:
    west, south, east, north = polygon.total_bounds
    plot_bbox = north, south, east, west
    return plot_bbox


def preprocess_features_building(features_data):
    # find nan level with 1

    if features_data.get("building:levels") is None:
        # add building:levels =1
        features_data["building:levels"] = 1
        return features_data
    features_data["building:levels"] = pd.to_numeric(
        features_data["building:levels"], errors="coerce"
    )
    nan_level = features_data["building:levels"].isna()
    # convert to float then round to int to remove decimal
    features_data.loc[nan_level, "building:levels"] = 1

    features_data["building:levels"] = (
        features_data["building:levels"].astype(float).round().astype(int)
    )

    # convert 0 to 1
    features_data.loc[features_data["building:levels"] == 0, "building:levels"] = 1
    # replace nan level with 1
    return features_data


def preprocess_features(features_data, tags):
    if tags.get("building") == True:
        return preprocess_features_building(features_data)
    else:
        raise ValueError("tags not found")
