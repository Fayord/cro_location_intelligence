import easyocr
from typing import List
import os
from tqdm import tqdm
import pandas as pd
import json


def extract_ocr(image_path: str) -> List[dict]:
    store_id = image_path.split("/")[-1].split(".")[0]
    reader = easyocr.Reader(["en", "th"], gpu=True)
    result = reader.readtext(image_path)
    result_list = []
    for r in result:
        bboxes = r[0]
        text = r[1]
        confidence = r[2]
        result_data = {
            "store_id": store_id,
            "bbox": bboxes,
            "text": text,
            "confidence": confidence,
        }
        result_list.append(result_data)

    return result_list


def main():

    # input_folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_poi_zoom_19/full_image"
    # input_folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_poi_zoom_19_new/poi_full_image"
    # main_folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven_poi_zoom_19_new/"

    input_folder = "/Users/ford/Documents/coding/cro_location_intelligence/notebook/data_7_eleven_2024_04_09_300m_poi/full_image"
    main_folder = "/Users/ford/Documents/coding/cro_location_intelligence/notebook/data_7_eleven_2024_04_09_300m_poi/"

    image_name_list = os.listdir(input_folder)
    all_store_list = []
    fail_list = []
    for image_name in tqdm(image_name_list):
        image_path = os.path.join(input_folder, image_name)
        try:

            result_list = extract_ocr(image_path)
            all_store_list.extend(result_list)
        except:
            fail_list.append(image_name)
    # convert to dataframe

    df = pd.DataFrame(all_store_list)
    # save df to csv
    df.to_csv(os.path.join(main_folder, "ocr_output.csv"), index=False)
    df.to_feather(os.path.join(main_folder, "ocr_output.feather"))
    # save fail list as json

    with open(os.path.join(main_folder, "fail_list.json"), "w") as f:
        json.dump(fail_list, f)


if __name__ == "__main__":
    main()
