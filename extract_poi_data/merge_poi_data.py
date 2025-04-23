import os
import pickle

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


import os
import pickle
import geopandas as gpd
import pandas as pd
import warnings
from tqdm import tqdm
import gc


def create_save_path(ori_save_path, partial_count):
    ori_save_file_name = os.path.basename(ori_save_path)
    ori_save_dir_name = os.path.dirname(ori_save_path)
    save_path = f"{ori_save_dir_name}/{ori_save_file_name[:-4]}_{partial_count:04}.pkl"
    return save_path


def consolidate_gdf_pickles(file_dir, ori_save_path, sample_size=-1, max_size=50_000):
    """
    Consolidate multiple GeoPandas DataFrames from pickle files with multiple geometry columns.

    Parameters:
    -----------
    file_dir : str
        Directory containing pickle files to be processed
    save_dir : str
        Directory where the consolidated file will be saved
    sample_size : int, optional
        Number of files to process. Default is -1 (process all files)

    Returns:
    --------
    consolidated_gdf : GeoDataFrame
        Consolidated GeoPandas DataFrame containing data from all processed pickle files
    """
    # Suppress specific warnings during processing
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Validate input directories
    if not os.path.exists(file_dir):
        raise ValueError(f"Input directory {file_dir} does not exist.")
    ori_save_dir_path = os.path.dirname(ori_save_path)
    os.makedirs(ori_save_dir_path, exist_ok=True)
    # Create save directory if it doesn't exist

    # Get list of pickle files
    pickle_files = [f for f in os.listdir(file_dir) if f.endswith(".pkl")]
    # sort by date of creation
    pickle_files.sort(key=lambda x: os.path.getmtime(os.path.join(file_dir, x)))

    # Adjust sample size if needed
    if sample_size == -1:
        sample_size = len(pickle_files)
    else:
        sample_size = min(sample_size, len(pickle_files))

    # Initialize tracking for columns and geometry columns
    all_columns = set()
    geometry_columns = set()
    consolidated_data = []

    # Process pickle files
    for file_name in tqdm(pickle_files[:sample_size]):
        try:
            file_path = os.path.join(file_dir, file_name)

            # Load the pickle file
            with open(file_path, "rb") as f:
                gdf = pickle.load(f)

            # Verify it's a GeoPandas DataFrame
            if not isinstance(gdf, gpd.GeoDataFrame):
                print(f"Skipping {file_name}: Not a GeoPandas DataFrame")
                continue

            # Identify geometry columns
            current_geometry_columns = [
                col
                for col in gdf.columns
                if isinstance(gdf[col], gpd.geoseries.GeoSeries)
                or (hasattr(gdf[col], "geom_type") and hasattr(gdf[col], "crs"))
            ]

            # Update tracked geometry columns
            geometry_columns.update(current_geometry_columns)

            # Create a de-fragmented copy
            gdf_copy = gdf.copy()

            # Store processed DataFrame
            consolidated_data.append(gdf_copy)

            # Collect all unique columns
            all_columns.update(gdf_copy.columns)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Prepare consolidated DataFrame
    if not consolidated_data:
        raise ValueError("No valid GeoPandas DataFrames found in the directory.")

    # Create a master DataFrame with all columns
    consolidated_columns = list(all_columns)

    # Prepare data with consistent columns
    aligned_dataframes = []
    file_count = 0
    partial_count = 1
    # for gdf in tqdm(consolidated_data):
    while len(consolidated_data) > 0:
        # pop first item
        gdf = consolidated_data.pop(0)
        save_path = create_save_path(ori_save_path, partial_count)
        # if save_path exist skip for max_size times
        if os.path.exists(save_path):
            file_count += 1
            if file_count == max_size:
                file_count = 0
                partial_count += 1
                continue

        # Create a DataFrame with all columns, filling missing columns with NaN
        aligned_df = pd.DataFrame(index=gdf.index, columns=consolidated_columns)

        # Copy existing columns
        for col in gdf.columns:
            aligned_df[col] = gdf[col]

        # Convert back to GeoDataFrame, preserving all geometry columns
        aligned_gdf = gpd.GeoDataFrame(aligned_df)

        # Restore geometry columns
        for geom_col in geometry_columns:
            if geom_col in gdf.columns:
                aligned_gdf[geom_col] = gdf[geom_col]

        aligned_dataframes.append(aligned_gdf)

        file_count += 1
        if file_count == max_size:
            file_count = 0
            # save to pickle
            # Concatenate all DataFrames
            consolidated_gdf = gpd.GeoDataFrame(
                pd.concat(aligned_dataframes, ignore_index=True)
            )

            # Ensure clean memory usage
            consolidated_gdf = consolidated_gdf.copy()

            # Save consolidated file
            consolidated_gdf.to_pickle(save_path)
            del consolidated_gdf  # Explicitly delete the object
            gc.collect()  # Trigger garbage collection

            aligned_dataframes = []
            partial_count += 1

    if file_count != 0:
        # Concatenate all DataFrames
        consolidated_gdf = gpd.GeoDataFrame(
            pd.concat(aligned_dataframes, ignore_index=True)
        )

        # Ensure clean memory usage
        consolidated_gdf = consolidated_gdf.copy()
        save_path = create_save_path(ori_save_path, partial_count)

        # Save consolidated file
        consolidated_gdf.to_pickle(save_path)
        del consolidated_gdf  # Explicitly delete the object
        gc.collect()  # Trigger garbage collection

    print(f"Consolidated {len(aligned_dataframes)} files.")
    print(f"Total partial_count: {partial_count}")
    print(f"Geometry columns preserved: {geometry_columns}")
    print(f"Saved to: {save_path}")

    return consolidated_gdf


# Example usage
# consolidated_df = consolidate_gdf_pickles(
#     file_dir='/path/to/pickle/files',
#     save_dir='/path/to/save/consolidated',
#     sample_size=-1
# )


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    poi_data_path = f"{dir_path}/data/2024_11_data_7_eleven_poi_data/poi_data"

    road_building_path = f"{dir_path}/data/2024_11_data_7_eleven/road_info"
    poi_data_list = os.listdir(poi_data_path)
    poi_data_path_list = [f"{poi_data_path}/{poi_data}" for poi_data in poi_data_list]
    # merge road and building data
    sample_size = -1
    # sample_size = 100

    save_path = f"{dir_path}/merge_data/merge_poi_data.pkl"
    merge_poi_df = consolidate_gdf_pickles(
        file_dir=poi_data_path,
        ori_save_path=save_path,
        sample_size=sample_size,
        # max_size=10,
    )
    print(merge_poi_df.shape)
    print(merge_poi_df.columns.to_list())


if __name__ == "__main__":
    main()
