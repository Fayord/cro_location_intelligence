# Run Sequence

To execute the sequence of scripts in order, follow the steps below:
## For POI Data
this data -> all_pois_relabeled_v4.feather have some duplicate
so I create a new column to have a unique_id

0. **Run Preprocess**
   ```bash
   python 0_pre_process_poi_data.py
   ```
1. **Run geo_poi_data**
   ```bash
   python geo_poi_data.py
   ```
   should set SAMPLE_DATA to 100 for testing first then set to -1 to run all 
   it used space around <= 35 GB from 1,642,327 row of poi data

1. **First plan is to Run merge_poi_data**
   ```bash
   dont python merge_poi_data.py
   ```
   but data is too much can't pack it in one file 
   so I read file and send it to iceberg instead

1. **Run dreamio_poi_data_batches**
   ```bash
   python dreamio_poi_data_batches.py
   ```
---

## For Road length and Building Area around 7-11

1. **Run geo_save_road_building**
   ```bash
   python geo_save_road_building.py
   ```
1. **Run merge_road_building**
   ```bash
   python merge_road_building.py
   ```
1. **Run dreamio_7_store**
   ```bash
   python dreamio_7_store.py
   ```




Make sure to have all dependencies installed for each script before running them.

