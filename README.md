# High-resolution maps of Arctic surface skin temperature and type retrieved from airborne thermal infrared imagery collected during the HALO–(AC)³ campaign
<<<<<<< HEAD
# High-resolution maps of Arctic surface skin temperature and type retrieved from airborne thermal infrared imagery collected during the HALO–(AC)³ campaign

This repository reproduces the figures, data and code outlined in Müller et al., (in submission to AMT). To produce the data, all scripts in the `run` directory need to be executed

      
run/
│
├── 00_autogluon_predict.ipynb
│   ├── **Input**: Autogluon model (e.g., `AutogluonModels/rf_HALO-AC3`)
│   ├── **Output**: Predictions for `BT` and roughness
│   └── **Purpose**: Setup AutoGluon model for downstream tasks.
│
├── 01_merge_VELOX_bands_and_write_to_disk.py
│   ├── **Input**: `base_filenames.csv` (location of all the individual datasets from the different bands, col=day and row=path_to_VELOX_data_of_band_i)
│   ├── **Output**: `{date}_v_0.1.zarr`
│   └── **Purpose**: Merges all VELOX bands into Zarr format for one date.
│
├── 02_predict_surface_type_and_calc_sst.py
│   ├── **Input**: `{date}_v_0.1.zarr` ← Output from `01`
│   ├── **Output**: `predicted_{time}.nc` files
│   ├── **Aux**: `flight_segments_HALO.csv` ← provided by `ac3airborne` package
│   └── **Purpose**: Predicts surface type and skin temperature using AutoGluon.
│
├── 03_make_pushbroom.py
│   ├── **Input**: `predicted_{time}.nc` ← Output from `02`
│   ├── **Aux**: `filenames.csv` for BAHAMAS data
│   ├── **Output**: `_concat.nc` files
│   └── **Purpose**: Produces pushbroom-concatenated images.
│
├── 04_merge_predictions.ipynb
│   ├── **Input**: `_concat.nc` ← Output from `03`
│   ├── **Output**: Merged predictions
│   └── **Purpose**: Merges predictions into single output files.
│
├── 05_segment_pushbroom.ipynb
│   ├── **Input**: `_concat.nc` ← Output from `03`
│   ├── **Output**: Segmented pushbroom files.
│   └── **Purpose**: Segments pushbroom images.
│
├── 06_get_segment_statistics.py
│   ├── **Input**: Segmented `_concat.nc` files ← Output from `05`
│   ├── **Output**: 
│   │   ├── `segmented_data_nadir/{key}.nc`
│   │   └── Full segmentation files (`*_full.nc`)
│   └── **Purpose**: Extracts statistics for each segment.
│
└── 07_process_segment_statistics_data.ipynb
    ├── **Input**: `segmented_data_nadir/{key}.nc` ← Output from `06`
    ├── **Output**: Processed segment statistics
    └── **Purpose**: Final processing and summarizing of segment statistics.