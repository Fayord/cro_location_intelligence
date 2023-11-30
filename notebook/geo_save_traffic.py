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
    time.sleep(2)

    # Take a screenshot of the entire page
    driver.save_screenshot(output_file)
    print(f"Screenshot saved to {output_file}")


from tqdm import tqdm
import os

if __name__ == "__main__":
    # Replace 'http://localhost:5173/' with your website URL
    website_url = "http://localhost:5173/"

    # Replace 'screenshot.png' with your desired output file name
    # data_path = "/Users/user/Documents/Coding/geo/notebook/7-11 Location for Ford.xlsx"

    project = "chester"
    project = "7-eleven"
    post_fix = "weekday_12"
    if project == "chester":
        data_path = (
            "/Users/user/Documents/Coding/geo/notebook/chester_branch_post_process.xlsx"
        )
        output_folder = f"/Users/user/Documents/Coding/geo/notebook/data_chester_{post_fix}/raw_image"
    elif project == "7-eleven":
        data_path = (
            "/Users/user/Documents/Coding/geo/notebook/7-11 Location for Ford.xlsx"
        )
        output_folder = f"/Users/user/Documents/Coding/geo/notebook/data_7_eleven_{post_fix}/raw_image"
    else:
        raise ValueError("project must be chester or 7-eleven")
    os.makedirs(output_folder, exist_ok=True)
    df = load_data_excel(data_path)
    # save as csv
    print(df)
    # loop each row
    # get lat lon
    driver = creat_driver(website_url)
    time.sleep(5)
    config_file_path = "/Users/user/Documents/Coding/js-samples/index.ts"
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        #     continue
        lat = row["latitude"]
        lon = row["longitude"]
        store_id = row["store_id"]
        update_latlon(config_file_path, lat, lon)
        output_file_name = os.path.join(output_folder, f"{store_id}.png")

        take_screenshot(driver, output_file_name)
        # if index > 5:
        #     break
        # take_screenshot(website_url, output_file_name)
    driver.quit()

    # take_screenshot(website_url, output_file_name)
