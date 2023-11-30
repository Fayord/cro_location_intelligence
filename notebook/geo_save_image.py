from geo_utils import (
    load_data_excel,
    save_full_image_from_lat_lon,
)


from tqdm import tqdm

import json
import time
import os


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = "/Users/user/Documents/Coding/geo/notebook/7-11 Location for Ford.xlsx"
    df = load_data_excel(data_path)
    # save as csv
    df.to_csv("7 lat long.csv", index=False)
    print(df)
    dpi = 600 // 3
    dist = 800
    edge_linewidth = dpi / 600 / 2
    fail_list = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        time.sleep(0.2)
        #     continue
        lat = row["latitude"]
        lon = row["longitude"]
        store_id = row["store_id"]
        # if store_id != 1262:
        #     continue
        # print(f"\n {index}, {store_id}")
        try:
            save_full_image_from_lat_lon(lat, lon, store_id, dist, dpi, edge_linewidth)
        except:
            fail_list.append(store_id)
        # break

    # save fail_list
    with open(f"{dir_path}/fail_list.json", "w") as f:
        json.dump(fail_list, f)


if __name__ == "__main__":
    main()
