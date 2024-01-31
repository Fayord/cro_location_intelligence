from PIL import Image
import os
from tqdm import tqdm
from geo_utils import crop_middle_image_from_side_size, apply_circle_mask

# HEIGHT_RESIZE_RATIO = 0.8240223463687151  # got from /Users/user/Documents/Coding/geo/notebook/geo_traffic_align.ipynb
HEIGHT_RESIZE_RATIO = 0.7631578947368421  # zoom 19 for P'how
HEIGHT_RESIZE_RATIO = 0.7631578947368421  # zoom 16.7 for N'kit


def get_full_image_size(
    image: Image.Image, height_resize_ratio: float = HEIGHT_RESIZE_RATIO
) -> tuple:
    _, height = image.size
    height = int(height * height_resize_ratio)
    return height, height


def crop_middle_image(image: Image.Image, full_image_size: tuple) -> Image.Image:
    width, height = image.size
    crop_width = int(height * full_image_size[0] / height)
    crop_height = int(height * full_image_size[0] / height)
    # crop image
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2
    image = image.crop((left, top, right, bottom))
    return image


def extract_full_image(image_path: str):
    image = Image.open(image_path)
    full_image_size = get_full_image_size(image)
    # crop image with full_image_size on the center
    image = crop_middle_image(image, full_image_size)
    return image


import numpy as np
from scipy.spatial import distance
import pandas as pd


def hex_to_rgb(hex_color):
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def extract_rgb_pixel_from_image(image: Image.Image, color_rgb_dict, color_cache: dict):
    image = image.convert("RGB")
    # Get the image data as a NumPy array
    img_array = np.array(image)

    # Get the dimensions of the image
    height, width, _ = img_array.shape

    # Initialize lists to store x, y, and color data
    x_values = []
    y_values = []
    color_values = []

    # Iterate through the image pixels
    for y in range(height):
        for x in range(width):
            # Get the RGB color values
            r, g, b = img_array[y, x]

            # Append x, y, and color values to the lists
            x_values.append(x)
            y_values.append(height - y - 1)
            color_values.append((r, g, b))

    # Create a DataFrame from the lists
    df = pd.DataFrame({"x": x_values, "y": y_values, "color": color_values})

    # Apply the function to create the 'nearest_color' column
    df["nearest_color"] = df["color"].apply(
        lambda x: find_nearest_color_with_cache(x, color_rgb_dict, color_cache)
    )

    return df


def find_nearest_color_with_cache(rgb_tuple, color_rgb_dict, color_cache):
    # Check if the color is already in the cache
    if rgb_tuple in color_cache:
        return color_cache[rgb_tuple]

    # If not in the cache, find the nearest color
    nearest_color = min(
        color_rgb_dict.keys(),
        key=lambda x: distance.euclidean(rgb_tuple, color_rgb_dict[x]),
    )

    # Update the cache
    color_cache[rgb_tuple] = nearest_color

    return nearest_color


def group_rgb_pixel_to_grid(rgb_pixel_df: pd.DataFrame, num_grid: int = 16):
    rgb_grid_df = rgb_pixel_df.copy()
    x_min = rgb_grid_df["x"].min()
    x_max = rgb_grid_df["x"].max()
    y_min = rgb_grid_df["y"].min()
    y_max = rgb_grid_df["y"].max()
    # want to group it into grid 16x16
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_spacing = x_range / (num_grid - 1)
    y_spacing = y_range / (num_grid - 1)
    grid_area = x_spacing * y_spacing
    rgb_grid_df["grid_lon_id"] = rgb_grid_df["x"].apply(
        lambda x: int((x + x_spacing / 2 - x_min) / x_spacing)
    )
    rgb_grid_df["grid_lat_id"] = rgb_grid_df["y"].apply(
        lambda y: int((y + y_spacing / 2 - y_min) / y_spacing)
    )

    # i want to count how many each color in each grid
    rgb_grid_df = (
        rgb_grid_df.groupby(["grid_lon_id", "grid_lat_id", "nearest_color"])
        .size()
        .reset_index(name="count")
    )
    # normalize count by grid_area
    rgb_grid_df["count"] = rgb_grid_df["count"] / grid_area
    pivot_df = rgb_grid_df.pivot_table(
        index=["grid_lon_id", "grid_lat_id"],
        columns="nearest_color",
        values="count",
        fill_value=0,
    ).reset_index()

    # pivot_df drop nearest_color
    # move nearest_color out of index
    pivot_df.columns.name = None
    # drop white and grey
    columns_to_drop = ["white", "grey"]
    columns_to_drop_existing = [
        col for col in columns_to_drop if col in pivot_df.columns
    ]
    pivot_df = pivot_df.drop(columns=columns_to_drop_existing, errors="ignore")
    return pivot_df


def process_image(
    image_file_name,
    image_folder_path,
    full_image_folder_path,
    crop_image_folder_path,
    raw_data_folder_path,
    grid_data_folder_path,
    color_rgb_dict,
    color_cache,
    IS_SAVE,
):
    save_grid_data_folder_path = os.path.join(
        grid_data_folder_path, image_file_name.replace(".png", ".csv")
    )
    if os.path.exists(save_grid_data_folder_path) is True:
        return
    if image_file_name.startswith("."):
        return
    image_path = os.path.join(image_folder_path, image_file_name)
    save_full_image_path = os.path.join(full_image_folder_path, image_file_name)
    if os.path.exists(save_full_image_path) is False:
        image = extract_full_image(image_path)
        if IS_SAVE is True:
            image.save(save_full_image_path)
    else:
        image = Image.open(save_full_image_path)
        
    ## for p how
    # save_crop_image_path = os.path.join(crop_image_folder_path, image_file_name)
    # if os.path.exists(save_crop_image_path) is False:
    #     # image = crop_middle_image_from_side_size(image, 2400, 1600)
    #     image = apply_circle_mask(image)
    #     if IS_SAVE is True:
    #         image.save(save_crop_image_path)
    # else:
    #     image = Image.open(save_crop_image_path)

    # # return
    # # save rgb_grid_data to csv
    # save_raw_data_folder_path = os.path.join(
    #     raw_data_folder_path, image_file_name.replace(".png", ".pkl")
    # )
    # if os.path.exists(save_raw_data_folder_path) is False:
    #     rgb_pixel_df = extract_rgb_pixel_from_image(image, color_rgb_dict, color_cache)
    #     if IS_SAVE is True:
    #         rgb_pixel_df.to_pickle(save_raw_data_folder_path)
    # else:
    #     rgb_pixel_df = pd.read_pickle(save_raw_data_folder_path)

    # if os.path.exists(save_grid_data_folder_path) is False:
    #     rgb_grid_df = group_rgb_pixel_to_grid(rgb_pixel_df)
    #     store_id = image_file_name.replace(".png", "")
    #     rgb_grid_df["store_id"] = store_id
    #     rgb_grid_df.to_csv(save_grid_data_folder_path)


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial


def main():
    IS_SAVE = True
    # IS_SAVE = False
    color_cache = {}

    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_weekday_12"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_weekday_18"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_weekend_12"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_weekend_18"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_chester_weekday_12"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_chester_weekday_18"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_chester_weekend_12"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_chester_weekend_18"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_zoom19"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_1"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_2"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_zoom16_7"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_zoom16_7_new"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_zoom19_new"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_terrain_missing_zoom19"
    # main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_terrain_missing_zoom16_7"
    main_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_poi_zoom_19"


    image_folder_path = f"{main_folder_path}/raw_image"
    full_image_folder_path = f"{main_folder_path}/full_image"
    crop_image_folder_path = f"{main_folder_path}/crop_image"
    raw_data_folder_path = f"{main_folder_path}/raw_data"
    grid_data_folder_path = f"{main_folder_path}/grid_data"
    print("main_folder_path", main_folder_path)
    os.makedirs(full_image_folder_path, exist_ok=True)
    os.makedirs(crop_image_folder_path, exist_ok=True)
    os.makedirs(raw_data_folder_path, exist_ok=True)
    os.makedirs(grid_data_folder_path, exist_ok=True)
    image_file_name_list = os.listdir(image_folder_path)
    # image_file_name_list = ["1025.png"]
    color_hex_dict = {
        "green": "82D375",
        "orange": "F19C5B",
        "red": "E04C3E",
        "maroon": "762824",
        "white": "FFFFFF",
        "grey": "E5E3E0",
    }
    color_rgb_dict = {key: hex_to_rgb(value) for key, value in color_hex_dict.items()}

    with ProcessPoolExecutor() as executor:
        process_func = partial(
            process_image,
            image_folder_path=image_folder_path,
            full_image_folder_path=full_image_folder_path,
            crop_image_folder_path=crop_image_folder_path,
            raw_data_folder_path=raw_data_folder_path,
            grid_data_folder_path=grid_data_folder_path,
            color_rgb_dict=color_rgb_dict,
            color_cache=color_cache,
            IS_SAVE=IS_SAVE,
        )
        list(
            tqdm(
                executor.map(process_func, image_file_name_list),
                total=len(image_file_name_list),
                desc="Processing images",
            )
        )


if __name__ == "__main__":
    # image_path = (
    #     "/Users/user/Documents/Coding/geo/notebook/data_chester_weekday_12/1025.png"
    # )
    # image = extract_full_image(image_path)
    # save_path = "/Users/user/Documents/Coding/geo/notebook/data_chester_weekday_12/1025_full.png"
    # image.save(save_path)
    main()
