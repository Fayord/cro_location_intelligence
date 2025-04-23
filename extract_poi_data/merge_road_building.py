import os
import pickle
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def consolidate_gdf_pickles_old(file_dir, save_path, sample_size=-1):
    """
    Consolidate multiple GeoPandas DataFrames from pickle files into a single DataFrame.

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
    # Validate input directories
    if not os.path.exists(file_dir):
        raise ValueError(f"Input directory {file_dir} does not exist.")

    # Create save directory if it doesn't exist

    # Get list of pickle files
    pickle_files = [f for f in os.listdir(file_dir) if f.endswith(".pkl")]

    # Adjust sample size if needed
    if sample_size == -1:
        sample_size = len(pickle_files)
    else:
        sample_size = min(sample_size, len(pickle_files))

    # Initialize list to store all DataFrames
    all_gdfs = []

    # Track all unique columns across all files
    all_columns = set()

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
            # verify gdf row!=0
            if gdf.shape[0] == 0:
                # print(f"Skipping {file_name}: Empty DataFrame")
                continue
            # Update all columns set
            all_columns.update(gdf.columns)

            # Append to list
            all_gdfs.append(gdf)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Ensure all DataFrames have all columns (fill with NaN if missing)
    consolidated_gdfs = []
    for gdf in all_gdfs:
        # Create a copy to avoid modifying original
        gdf_copy = gdf.copy()

        # Add missing columns with NaN
        missing_columns = all_columns - set(gdf_copy.columns)
        for col in missing_columns:
            gdf_copy[col] = pd.NA

        consolidated_gdfs.append(gdf_copy)

    # Concatenate all DataFrames
    if not consolidated_gdfs:
        raise ValueError("No valid GeoPandas DataFrames found in the directory.")

    consolidated_gdf = gpd.GeoDataFrame(pd.concat(consolidated_gdfs, ignore_index=True))

    # Save consolidated file
    consolidated_gdf.to_pickle(save_path)

    print(f"Consolidated {len(consolidated_gdfs)} files.")
    print(f"Total columns: {len(all_columns)}")
    print(f"Saved to: {save_path}")

    return consolidated_gdf


import os
import pickle
import geopandas as gpd
import pandas as pd
import warnings
from tqdm import tqdm


def consolidate_gdf_pickles(file_dir, save_path, sample_size=-1):
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

    # Create save directory if it doesn't exist

    # Get list of pickle files
    pickle_files = [f for f in os.listdir(file_dir) if f.endswith(".pkl")]

    # Adjust sample size if needed
    if sample_size == -1:
        sample_size = len(pickle_files)
    else:
        sample_size = min(sample_size, len(pickle_files))

    # Initialize tracking for columns and geometry columns
    all_columns = set()
    geometry_columns = set()
    consolidated_data = []
    remove_columns = ["geometry"]
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
            # remove column in remove_columns
            for col in remove_columns:
                if col in gdf.columns:
                    gdf = gdf.drop(col, axis=1)
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
    for gdf in tqdm(consolidated_data):
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

    # Concatenate all DataFrames
    consolidated_gdf = gpd.GeoDataFrame(
        pd.concat(aligned_dataframes, ignore_index=True)
    )

    # Ensure clean memory usage
    consolidated_gdf = consolidated_gdf.copy()

    # Save consolidated file
    consolidated_gdf.to_pickle(save_path)

    print(f"Consolidated {len(aligned_dataframes)} files.")
    print(f"Total columns: {len(all_columns)}")
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

    save_path = f"{dir_path}/merge_road_building_data.pkl"
    # merge_road_building_df = consolidate_gdf_pickles_old(
    merge_road_building_df = consolidate_gdf_pickles(
        file_dir=road_building_path,
        save_path=save_path,
        sample_size=sample_size,
    )
    print(merge_road_building_df.shape)
    print(merge_road_building_df.columns.to_list())


if __name__ == "__main__":
    main()
