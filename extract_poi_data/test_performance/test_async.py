import os
import json
import time
import asyncio
from random import randint
from pathlib import Path
import aiofiles

# Directory for mock JSON files
MOCK_DATA_DIR = "mock_data"
MERGED_FILE = "merged_data.json"
MERGED_FILE_ASYNC = "merged_data_async.json"
NUM_FILES = 100
RECORDS_PER_FILE = 10000


# Function to generate mock data files
def generate_mock_data():
    os.makedirs(MOCK_DATA_DIR, exist_ok=True)
    for i in range(NUM_FILES):
        data = [{"id": j, "value": randint(1, 100)} for j in range(RECORDS_PER_FILE)]
        with open(os.path.join(MOCK_DATA_DIR, f"data_{i}.json"), "w") as f:
            json.dump(data, f)
    print(f"Generated {NUM_FILES} mock JSON files.")


# Synchronous function to read and preprocess a file
def read_and_preprocess_sync(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return [{"id": record["id"], "value": record["value"] * 2} for record in data]


# Asynchronous function to read and preprocess a file
async def read_and_preprocess_async(filepath):
    async with aiofiles.open(filepath, "r") as f:
        data = json.loads(await f.read())
    return [{"id": record["id"], "value": record["value"] * 2} for record in data]


# Synchronous implementation
def process_sync():
    start_time = time.time()
    merged_data = []
    for filepath in Path(MOCK_DATA_DIR).glob("*.json"):
        merged_data.extend(read_and_preprocess_sync(filepath))
    with open(MERGED_FILE, "w") as f:
        json.dump(merged_data, f)
    print(f"Sync: Processed and merged in {time.time() - start_time:.2f} seconds.")


# Asynchronous implementation
async def process_async():
    import aiofiles

    start_time = time.time()
    tasks = [
        read_and_preprocess_async(filepath)
        for filepath in Path(MOCK_DATA_DIR).glob("*.json")
    ]
    results = await asyncio.gather(*tasks)
    merged_data = [item for sublist in results for item in sublist]
    async with aiofiles.open(MERGED_FILE_ASYNC, "w") as f:
        await f.write(json.dumps(merged_data))
    print(f"Async: Processed and merged in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    # Generate mock data if not already generated
    if not os.path.exists(MOCK_DATA_DIR):
        generate_mock_data()

    print("Running synchronous processing:")
    process_sync()

    print("\nRunning asynchronous processing:")
    asyncio.run(process_async())
