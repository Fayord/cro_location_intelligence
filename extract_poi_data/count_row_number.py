import os
import pandas as pd
import asyncio
import time


async def read_pickle_files(dir_folder, max_files=None):
    """
    Reads all .pkl files in the given directory asynchronously and returns the total row count.
    If max_files is provided, only processes up to that many files for benchmarking.
    """
    total_rows = 0
    processed_files = 0

    async def process_file(file_path):
        nonlocal total_rows, processed_files
        df = pd.read_pickle(file_path)
        total_rows += len(df)
        processed_files += 1

    tasks = []
    for i, file_name in enumerate(os.listdir(dir_folder)):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(dir_folder, file_name)
            tasks.append(process_file(file_path))
            if max_files and len(tasks) >= max_files:
                break

    await asyncio.gather(*tasks)

    return total_rows, processed_files


# Example usage
async def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_folder = f"{dir_path}/data/2024_11_data_7_eleven_poi_data/poi_data"
    benchmark_files = 1000  # Number of files for benchmarking

    # Benchmarking
    start_time = time.time()
    _, processed_files = await read_pickle_files(dir_folder, max_files=benchmark_files)
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_per_file = elapsed_time / processed_files
    estimated_total_time = (
        avg_time_per_file * 1_600_000
    )  # Estimate for 1.6 million files

    print(f"Processed {processed_files} files in {elapsed_time:.2f} seconds.")
    print(f"Average time per file: {avg_time_per_file:.4f} seconds")
    print(
        f"Estimated total time for 1.6 million files: {estimated_total_time / 3600:.2f} hours"
    )


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
