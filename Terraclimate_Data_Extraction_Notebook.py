import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import pystac_client
import planetary_computer as pc
from tqdm import tqdm
import logging

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('terraclimate_extraction.log'),
        logging.StreamHandler()
    ]
)

print("✅ Dependencies loaded successfully!")

# ── Variables to extract ──────────────────────────────────────────────────────
# pet  = potential evapotranspiration (original)
# aet  = actual evapotranspiration
# def  = climate water deficit
# ppt  = precipitation
# q    = runoff
# soil = soil moisture
# pdsi = Palmer Drought Severity Index
# tmax = max temperature
# tmin = min temperature
# vap  = vapor pressure
# swe  = snow water equivalent

VARIABLES = ['pet', 'aet', 'def', 'ppt', 'q', 'soil', 'pdsi', 'tmax', 'tmin', 'vap', 'swe']

# South Africa bounding box
LAT_MIN, LAT_MAX = -35.18, -21.72
LON_MIN, LON_MAX =  14.97,  32.79


# ── Load dataset ──────────────────────────────────────────────────────────────

def load_terraclimate_dataset():
    logging.info("Loading TerraClimate dataset from Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )

    logging.info(f"Dataset loaded. Variables available: {list(ds.data_vars)}")
    return ds


# ── Filter to SA region and time range ───────────────────────────────────────

def filterg(ds, var):
    logging.info(f"Filtering {var} for 2011-2015 over South Africa...")
    ds_filtered = ds[var].sel(time=slice("2011-01-01", "2015-12-31"))

    df_var_append = []
    for i in tqdm(range(len(ds_filtered.time)), desc=f"Filtering {var}"):
        df_var = ds_filtered.isel(time=i).to_dataframe().reset_index()
        df_var_filter = df_var[
            (df_var['lat'] > LAT_MIN) & (df_var['lat'] < LAT_MAX) &
            (df_var['lon'] > LON_MIN) & (df_var['lon'] < LON_MAX)
        ]
        df_var_append.append(df_var_filter)

    df_var_final = pd.concat(df_var_append, ignore_index=True)
    df_var_final['time'] = df_var_final['time'].astype(str)
    df_var_final = df_var_final.rename(columns={
        "lat": "Latitude",
        "lon": "Longitude",
        "time": "Sample Date"
    })

    logging.info(f"Filtering for {var} completed: {df_var_final.shape}")
    return df_var_final


# ── Map nearest climate values to sample locations ────────────────────────────

def assign_nearest_climate(sa_df, climate_df, var_name):
    logging.info(f"Mapping {var_name} to sample locations...")

    sa_coords      = np.radians(sa_df[['Latitude', 'Longitude']].values)
    climate_coords = np.radians(climate_df[['Latitude', 'Longitude']].values)

    tree = cKDTree(climate_coords)
    _, idx = tree.query(sa_coords, k=1)

    nearest_points = climate_df.iloc[idx].reset_index(drop=True)
    sa_df = sa_df.reset_index(drop=True)
    sa_df[['nearest_lat', 'nearest_lon']] = nearest_points[['Latitude', 'Longitude']]

    sa_df['Sample Date']      = pd.to_datetime(sa_df['Sample Date'],      dayfirst=True, errors='coerce')
    climate_df['Sample Date'] = pd.to_datetime(climate_df['Sample Date'], dayfirst=True, errors='coerce')

    climate_values = []
    for i in tqdm(range(len(sa_df)), desc=f"Mapping {var_name.upper()}"):
        sample_date  = sa_df.loc[i, 'Sample Date']
        nearest_lat  = sa_df.loc[i, 'nearest_lat']
        nearest_lon  = sa_df.loc[i, 'nearest_lon']

        subset = climate_df[
            (climate_df['Latitude']  == nearest_lat) &
            (climate_df['Longitude'] == nearest_lon)
        ]

        if subset.empty:
            climate_values.append(np.nan)
            continue

        nearest_idx = (subset['Sample Date'] - sample_date).abs().idxmin()
        climate_values.append(subset.loc[nearest_idx, var_name])

    return climate_values


# ── Extract all variables for a dataset ──────────────────────────────────────

def extract_all_variables(df, ds, label="dataset"):
    logging.info(f"Extracting {len(VARIABLES)} TerraClimate variables for {label}...")
    results = df[['Latitude', 'Longitude', 'Sample Date']].copy()

    for var in VARIABLES:
        if var not in ds.data_vars:
            logging.warning(f"Variable {var} not found in dataset, skipping.")
            results[var] = np.nan
            continue

        try:
            climate_df     = filterg(ds, var)
            climate_values = assign_nearest_climate(df.copy(), climate_df, var)
            results[var]   = climate_values
            logging.info(f"✅ {var} extracted successfully")
        except Exception as e:
            logging.error(f"❌ Failed to extract {var}: {e}")
            results[var] = np.nan

    return results


# ── Derived climate features ──────────────────────────────────────────────────

def add_derived_features(df):
    logging.info("Computing derived climate features...")

    # Temperature range — proxy for continentality / thermal stress
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['temp_range'] = df['tmax'] - df['tmin']

    # Aridity index — ratio of precip to PET, low = arid
    if 'ppt' in df.columns and 'pet' in df.columns:
        df['aridity_index'] = df['ppt'] / (df['pet'] + 1e-10)

    # Evaporative fraction — how much of PET is actually met
    if 'aet' in df.columns and 'pet' in df.columns:
        df['evap_fraction'] = df['aet'] / (df['pet'] + 1e-10)

    # Water surplus/deficit — positive means surplus
    if 'ppt' in df.columns and 'aet' in df.columns:
        df['water_balance'] = df['ppt'] - df['aet']

    logging.info(f"Derived features added. Final shape: {df.shape}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load dataset once — reused for both training and validation
    ds = load_terraclimate_dataset()

    # ── Training data ──
    logging.info("Processing training dataset...")
    Water_Quality_df = pd.read_csv('water_quality_training_dataset.csv')
    logging.info(f"Training dataset: {Water_Quality_df.shape}")

    train_results = extract_all_variables(Water_Quality_df, ds, label="training")
    train_results = add_derived_features(train_results)
    train_results.to_csv('terraclimate_features_training.csv', index=False)

    logging.info(f"✅ Training extraction complete: {train_results.shape}")
    print(train_results.head(3))

    # ── Validation data ──
    import os
    val_file = 'submission_template.csv'
    logging.info(f"Processing validation dataset from {val_file}...")

    Validation_df = pd.read_csv(val_file)
    logging.info(f"Validation dataset: {Validation_df.shape}")

    val_results = extract_all_variables(Validation_df, ds, label="validation")
    val_results = add_derived_features(val_results)
    val_results.to_csv('terraclimate_features_validation.csv', index=False)

    logging.info(f"✅ Validation extraction complete: {val_results.shape}")
    print(val_results.head(3))