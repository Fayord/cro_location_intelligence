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


def update_traffic_df(traffic_df, postfix):
    # drop Unnamed: 0
    select_columns = [
        "green",
        "orange",
        "red",
        "maroon",
    ]
    traffic_df = traffic_df.drop(columns=["Unnamed: 0"])
    for select_column in select_columns:
        try:
            traffic_df[f"{select_column}{postfix}"] = traffic_df[select_column]
        except KeyError:
            traffic_df[f"{select_column}{postfix}"] = 0.0
            traffic_df[f"{select_column}"] = 0.0
    # drop select_columns
    traffic_df = traffic_df.drop(columns=select_columns)
    return traffic_df


from tqdm import tqdm


def main():
    project = "chester"
    project = "7-eleven"
    if project == "chester":
        dir_folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_chester"
        save_destination_folder_path = f"{dir_folder}"
        save_destination_file_path = os.path.join(
            save_destination_folder_path, "data_chester.pkl"
        )
    elif project == "7-eleven":
        dir_folder = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data_7_eleven"
        save_destination_folder_path = f"{dir_folder}"
        save_destination_file_path = os.path.join(
            save_destination_folder_path, "data_7_eleven.pkl"
        )
    else:
        raise ValueError("project must be chester or 7-eleven")

    road_info_folder_path = f"{dir_folder}/road_info"
    traffic_wkday_12_folder_path = f"{dir_folder}_weekday_12/grid_data"
    traffic_wkday_18_folder_path = f"{dir_folder}_weekday_18/grid_data"
    traffic_wkend_12_folder_path = f"{dir_folder}_weekend_12/grid_data"
    traffic_wkend_18_folder_path = f"{dir_folder}_weekend_18/grid_data"

    road_info_file_path_list = get_file_path_list(road_info_folder_path)
    # print(road_info_file_path_list[:5])
    traffic_wkday_12_file_path_list = get_file_path_list(traffic_wkday_12_folder_path)
    # print(traffic_wkday_12_file_path_list[:5])
    assert len(road_info_file_path_list) == len(traffic_wkday_12_file_path_list)

    # create a blank dataframe
    all_merged_df = pd.DataFrame()
    for road_info_file_path in tqdm(road_info_file_path_list, desc="Processing rows"):
        store_id = road_info_file_path.split("/")[-1].split(".")[0]
        traffic_wkday_12_file_path = os.path.join(
            traffic_wkday_12_folder_path, f"{store_id}.csv"
        )
        traffic_wkday_18_file_path = os.path.join(
            traffic_wkday_18_folder_path, f"{store_id}.csv"
        )
        traffic_wkend_12_file_path = os.path.join(
            traffic_wkend_12_folder_path, f"{store_id}.csv"
        )
        traffic_wkend_18_file_path = os.path.join(
            traffic_wkend_18_folder_path, f"{store_id}.csv"
        )
        assert os.path.exists(traffic_wkday_12_file_path)
        road_info_df = pd.read_pickle(road_info_file_path)
        traffic_wkday_12_df = pd.read_csv(traffic_wkday_12_file_path)
        traffic_wkday_18_df = pd.read_csv(traffic_wkday_18_file_path)
        traffic_wkend_12_df = pd.read_csv(traffic_wkend_12_file_path)
        traffic_wkend_18_df = pd.read_csv(traffic_wkend_18_file_path)
        traffic_wkday_12_df = update_traffic_df(traffic_wkday_12_df, "_wkday_12")
        traffic_wkday_18_df = update_traffic_df(traffic_wkday_18_df, "_wkday_18")
        traffic_wkend_12_df = update_traffic_df(traffic_wkend_12_df, "_wkend_12")
        traffic_wkend_18_df = update_traffic_df(traffic_wkend_18_df, "_wkend_18")

        traffic_df_list = [
            traffic_wkday_12_df,
            traffic_wkday_18_df,
            traffic_wkend_12_df,
            traffic_wkend_18_df,
        ]
        merged_df = road_info_df.copy()
        for traffic_df in traffic_df_list:
            merged_df = pd.merge(
                merged_df,
                traffic_df,
                on=["grid_lat_id", "grid_lon_id", "store_id"],
                how="left",
                validate="one_to_one",
            )

        # print("merged_df.shape", merged_df.shape)
        assert (
            merged_df.shape[0]
            == road_info_df.shape[0]
            == traffic_wkday_12_df.shape[0]
            == traffic_wkday_18_df.shape[0]
            == traffic_wkend_12_df.shape[0]
            == traffic_wkend_18_df.shape[0]
        ), f"road_info_file_path {road_info_file_path}"
        all_merged_df = pd.concat([all_merged_df, merged_df])
    all_merged_df = all_merged_df.fillna(0)

    # all_merged_df.drop(columns=["grid_id"], inplace=True)
    # re order columns
    all_columns = all_merged_df.columns.tolist()
    all_columns.remove("store_id")
    all_columns.remove("grid_lat_id")
    all_columns.remove("grid_lon_id")
    post_fix_list = [
        "_wkday_12",
        "_wkday_18",
        "_wkend_12",
        "_wkend_18",
    ]
    color_list = [
        "green",
        "orange",
        "red",
        "maroon",
    ]
    remove_columns = []
    for post_fix in post_fix_list:
        for color in color_list:
            remove_column = f"{color}{post_fix}"
            remove_columns.append(remove_column)
            all_columns.remove(remove_column)
    all_columns.remove("geometry")

    # to sort columns
    all_columns = (
        ["store_id", "grid_lat_id", "grid_lon_id"]
        + remove_columns
        + all_columns
        + ["geometry"]
    )
    all_merged_df = all_merged_df[all_columns]

    # fill null
    # save as pickle
    all_merged_df.to_pickle(save_destination_file_path, protocol=4)


if __name__ == "__main__":
    main()
