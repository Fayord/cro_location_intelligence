import easyocr
import os
import json
from tqdm import tqdm
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context


def clean_results(results):
    new_results = []
    for bboxes, text, confident in results:
        new_result = {}
        new_bboxes = []
        for bbox in bboxes:
            new_bboxes.append([int(x) for x in bbox])  # convert to int, it is pixel
        new_result["bbox"] = new_bboxes
        new_result["text"] = text
        new_result["confident"] = confident
        new_results.append(new_result)
    return new_results


def main():
    # main_folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_poi_zoom_19/"
    # folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_google_map_poi_zoom_19/raw_image"
    main_folder = "/Users/ford/Documents/coding/cro_location_intelligence/notebook/data_7_eleven_2024_03_15_300m_poi"
    folder = "/Users/ford/Documents/coding/cro_location_intelligence/notebook/data_7_eleven_2024_03_15_300m_poi/full_image"

    image_name_list = os.listdir(folder)
    image_name_list = [x for x in image_name_list if not x.startswith(".")]

    fail_list = []
    poi_list = []
    for image_name in tqdm(image_name_list):
        # store_id = image_name.split(".")[0]
        # image_path = f"{folder}/{image_name}"
        # reader = easyocr.Reader(["en", "th"])
        # results = reader.readtext(image_path)
        # # convert third element to float
        # results = clean_results(results)
        # for result in results:
        #     bbox = result["bbox"]
        #     text = result["text"]
        #     confident = result["confident"]
        #     poi_data = {
        #         "store_id": store_id,
        #         "bbox": bbox,
        #         "text": text,
        #         "confidence": confident,
        #     }
        #     poi_list.append(poi_data)
        # break
        try:
            store_id = image_name.split(".")[0]
            image_path = f"{folder}/{image_name}"
            reader = easyocr.Reader(["en", "th"])
            results = reader.readtext(image_path)
            # convert third element to float
            results = clean_results(results)
            for result in results:
                bbox = result["bbox"]
                text = result["text"]
                confident = result["confident"]
                poi_data = {
                    "store_id": store_id,
                    "bbox": bbox,
                    "text": text,
                    "confidence": confident,
                }
                poi_list.append(poi_data)
        except:
            fail_list.append(image_name)
    # save fail list to main_folder
    with open(f"{main_folder}/fail_list.json", "w") as f:
        json.dump(fail_list, f)
    poi_df = pd.DataFrame(poi_list)
    # save poi_df to main_folder save to feather
    poi_df.to_feather(f"{main_folder}/poi_df.feather")


if __name__ == "__main__":
    main()
