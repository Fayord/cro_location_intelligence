import pandas as pd
import os
import hashlib


# Function to generate a hash-based unique ID
def generate_unique_id(row):
    # Combine the relevant fields into a single string
    identifier = f"{row['latitude']}_{row['longitude']}_{row['name']}"
    # Use a hash function (e.g., SHA-256) for a unique ID
    return hashlib.sha256(identifier.encode()).hexdigest()


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = f"{dir_path}/all_pois_relabeled_v4.feather"

    # show all columns
    df = pd.read_feather(data_path)
    print(df.columns)
    # Apply the function to each row
    df["unique_id"] = df.apply(generate_unique_id, axis=1)

    # Display the DataFrame
    # save as feather
    # all_pois_relabeled_v4_with_unique_id
    df.to_feather(f"{dir_path}/all_pois_relabeled_v4_with_unique_id.feather")


if __name__ == "__main__":
    main()
