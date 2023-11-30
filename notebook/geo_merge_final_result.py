import os
import pandas as pd


def get_file_path_list(folder_path):
    file_path_list = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith("."):
            continue
        file_path = os.path.join(folder_path, file_name)
        file_path_list.append(file_path)
    return file_path_list


def main():
    road_info_folder_path = (
        "/Users/user/Documents/Coding/geo/notebook/data_chester/road_info"
    )
    traffic_wkday_12_folder_path = (
        "/Users/user/Documents/Coding/geo/notebook/data_chester_weekday_12/grid_data"
    )
    save_destination_folder_path = (
        "/Users/user/Documents/Coding/geo/notebook/data_chester"
    )
    save_destination_file_path = os.path.join(
        save_destination_folder_path, "data_chester.pkl"
    )
    road_info_file_path_list = get_file_path_list(road_info_folder_path)
    # print(road_info_file_path_list[:5])
    traffic_wkday_12_file_path_list = get_file_path_list(traffic_wkday_12_folder_path)
    # print(traffic_wkday_12_file_path_list[:5])
    assert len(road_info_file_path_list) == len(traffic_wkday_12_file_path_list)

    # create a blank dataframe
    all_merged_df = pd.DataFrame()
    for road_info_file_path in road_info_file_path_list:
        store_id = road_info_file_path.split("/")[-1].split(".")[0]
        traffic_wkday_12_file_path = os.path.join(
            traffic_wkday_12_folder_path, f"{store_id}.csv"
        )
        assert os.path.exists(traffic_wkday_12_file_path)
        road_info_df = pd.read_pickle(road_info_file_path)
        traffic_wkday_12_df = pd.read_csv(traffic_wkday_12_file_path)
        print("road_info_df.shape", road_info_df.shape)
        print("traffic_wkday_12_df.shape", traffic_wkday_12_df.shape)
        merged_df = pd.merge(
            road_info_df,
            traffic_wkday_12_df,
            on=["grid_lat_id", "grid_lon_id", "store_id"],
            how="left",
            validate="one_to_one",
        )
        # remove Unnamed: 0
        merged_df = merged_df.drop(columns=["Unnamed: 0"])
        # fill nan with 0

        print("merged_df.shape", merged_df.shape)
        assert (
            merged_df.shape[0] == road_info_df.shape[0] == traffic_wkday_12_df.shape[0]
        ), f"road_info_file_path {road_info_file_path}"
        all_merged_df = pd.concat([all_merged_df, merged_df])
    all_merged_df = all_merged_df.fillna(0)

    all_merged_df.drop(columns=["grid_id"], inplace=True)
    # re order columns
    all_columns = all_merged_df.columns.tolist()
    all_columns.remove("store_id")
    all_columns.remove("grid_lat_id")
    all_columns.remove("grid_lon_id")
    all_columns.remove("green")
    all_columns.remove("orange")
    all_columns.remove("red")
    all_columns.remove("maroon")
    all_columns.remove("geometry")
    all_columns = (
        ["store_id", "grid_lat_id", "grid_lon_id", "green", "orange", "red", "maroon"]
        + all_columns
        + ["geometry"]
    )
    all_merged_df = all_merged_df[all_columns]

    # fill null
    # save as pickle
    all_merged_df.to_pickle(save_destination_file_path, protocol=4)


if __name__ == "__main__":
    main()
