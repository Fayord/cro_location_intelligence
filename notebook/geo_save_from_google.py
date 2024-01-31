from selenium import webdriver
import time

import pandas as pd
from geo_utils import load_data_excel


# Function to update latlon values in the .ts file
def update_latlon(file_path, new_lat, new_lon):
    with open(file_path, "r") as file:
        data = file.read()

    # Assuming the latlon values are stored as variables lat and lon in the .ts file
    # find line that have "center" remove
    data_split = data.split("\n")
    for i, line in enumerate(data_split):
        if "center" in line:
            data_split[i] = f"center: {{ lat: {new_lat}, lng: {new_lon} }},"
            break
    data = "\n".join(data_split)

    # then insert that line with center: { lat: lat, lng: lng },

    with open(file_path, "w") as file:
        file.write(data)


def creat_driver(url):
    # Set up the Chrome WebDriver with --disable-site-isolation-trials
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-site-isolation-trials")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver


def take_screenshot(driver, output_file):
    # driver.refresh()
    # Wait for the page to load (adjust the sleep time as needed)

    # Take a screenshot of the entire page
    driver.save_screenshot(output_file)


def divide_dataframe(df, total_part=4):
    """
    Divides a DataFrame into four equal parts and assigns each part an index (1, 2, 3, 4).

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing subsets of the DataFrame with assigned indices.
    """
    # Get the number of rows in the DataFrame
    num_rows = len(df)

    # Calculate the number of rows per part
    rows_per_part = num_rows // total_part

    # Initialize an empty dictionary to store subsets
    divided_data = {}

    # Divide the DataFrame into parts
    for i in range(total_part):
        start_index = i * rows_per_part
        stop_index = (i + 1) * rows_per_part if i < total_part - 1 else num_rows
        subset_df = df.iloc[start_index:stop_index]
        divided_data[i + 1] = subset_df  # Assigning index (1, 2, ..., total_part)

    return divided_data


from tqdm import tqdm
import os
import sys

if __name__ == "__main__":
    # Replace 'http://localhost:5173/' with your website URL
    # get port from sys
    port = sys.argv[1]
    num_part = int(sys.argv[2])
    website_url = f"http://localhost:{port}/"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Replace 'screenshot.png' with your desired output file name
    # data_path = "{dir_path}/7-11 Location for Ford.xlsx"

    # project = "chester"
    project = "7-eleven"
    # post_fix = "weekday_12"
    # post_fix = "google_map"
    # post_fix = f"google_map_{num_part}"
    post_fix = f"google_map_poi_zoom_19"
    if project == "chester":
        data_path = f"{dir_path}/chester_branch_post_process.xlsx"
        output_folder = f"{dir_path}/data_chester_{post_fix}/raw_image"
    elif project == "7-eleven":
        # data_path = f"{dir_path}/7-11 Location for Ford.xlsx"
        # data_path = f"{dir_path}/7-11 Location for Ford only_new.xlsx"
        # data_path = f"{dir_path}/7-11 Location for Ford missing_p_how.xlsx"
        data_path = f"{dir_path}/7-11 Location for Ford_all+missing.xlsx"
        # data_path = f"{dir_path}/ร้านใหม่(3).csv"

        output_folder = f"{dir_path}/data_7_eleven_{post_fix}/raw_image"
    else:
        raise ValueError("project must be chester or 7-eleven")
    os.makedirs(output_folder, exist_ok=True)
    try:
        df = load_data_excel(data_path)
    except:
        df = pd.read_csv(data_path)
    # save as csv
    print(df)
    # number of rows
    divided_df = divide_dataframe(df, total_part=2)
    df = divided_df[num_part]
    # loop each row
    # get lat lon
    driver = creat_driver(website_url)
    time.sleep(5)
    if port == "5173":
        config_file_path = "/Users/user/Documents/Coding/js-samples/index.ts"
    elif port == "5174":
        config_file_path = "/Users/user/Documents/Coding/js-samples copy/index.ts"
    else:
        raise ValueError("port must be 5173 or 5174")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        #     continue
        lat = row["latitude"]
        lon = row["longitude"]
        store_id = row["store_id"]
        update_latlon(config_file_path, lat, lon)
        output_file_name = os.path.join(output_folder, f"{store_id}.png")
        # time.sleep(1.2)
        # if output_file_name is exist skip
        if os.path.exists(output_file_name):
            print(f"skip {store_id}")
            continue
        time.sleep(1.2)
        take_screenshot(driver, output_file_name)
        # if index > 5:
        #     break
        # take_screenshot(website_url, output_file_name)
    driver.quit()

    # take_screenshot(website_url, output_file_name)
