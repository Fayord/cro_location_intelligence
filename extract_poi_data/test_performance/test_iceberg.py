import time
import json
from pyiceberg.catalog import Catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StructType, LongType, DoubleType
from pyiceberg.expressions import Expressions

# Iceberg catalog configuration
CATALOG_NAME = "local_catalog"
TABLE_NAME = "default.merge_data_table"
CATALOG_CONFIG = {"type": "hadoop", "warehouse": "file:///tmp/iceberg_warehouse"}


# Create a PyIceberg catalog
def create_catalog():
    return Catalog.create(CATALOG_NAME, CATALOG_CONFIG)


# Create a table schema
def create_table(catalog):
    schema = Schema(StructType().field("id", LongType()).field("value", DoubleType()))
    return catalog.create_table(
        name=TABLE_NAME,
        schema=schema,
        partition_spec=[],
        location=f"file:///tmp/iceberg_warehouse/{TABLE_NAME}",
    )


# Load merged data from the JSON file
def load_merged_data(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


# Write each row individually
def write_rows_individually(table, data):
    start_time = time.time()
    with table.new_append() as appender:
        for row in data:
            appender.append(row)
    duration = time.time() - start_time
    print(f"Wrote {len(data)} rows individually in {duration:.2f} seconds.")
    return duration


# Write all rows at once
def write_rows_bulk(table, data):
    start_time = time.time()
    with table.new_append() as appender:
        appender.append(data)
    duration = time.time() - start_time
    print(f"Wrote {len(data)} rows at once in {duration:.2f} seconds.")
    return duration


# Delete the table
def delete_table(catalog):
    catalog.drop_table(TABLE_NAME)
    print(f"Table {TABLE_NAME} deleted.")


if __name__ == "__main__":
    # Initialize catalog
    catalog = create_catalog()

    # Create the table
    try:
        table = create_table(catalog)
    except Exception as e:
        print(f"Error creating table: {e}")

    # Load merged data
    merged_data = load_merged_data("merged_data.json")

    # Convert JSON to rows in the format Iceberg expects
    iceberg_data = [
        {"id": record["id"], "value": float(record["value"])} for record in merged_data
    ]

    # Write each row individually
    duration_individual = write_rows_individually(table, iceberg_data)

    # Write all rows at once
    duration_bulk = write_rows_bulk(table, iceberg_data)

    # Print comparison
    print(f"\nPerformance Comparison:")
    print(f"Individual write duration: {duration_individual:.2f} seconds")
    print(f"Bulk write duration: {duration_bulk:.2f} seconds")

    # Delete the table
    delete_table(catalog)
