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
import os

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

# ── Config ────────────────────────────────────────────────────────────────────

VARIABLES = ['pet', 'aet', 'def', 'ppt', 'q', 'soil', 'pdsi', 'tmax', 'tmin', 'vap', 'swe']

LAT_MIN, LAT_MAX = -35.18, -21.72
LON_MIN, LON_MAX =  14.97,  32.79


# ── Load dataset (called fresh per variable to avoid token expiry) ────────────

def load_terraclimate_dataset():
    logging.info("Loading TerraClimate dataset (fresh token)...")
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
    return ds


# ── Fast spatial+temporal filter ─────────────────────────────────────────────

def filterg(ds, var):
    logging.info(f"Filtering {var} for 2011-2015 over South Africa...")

    # Slice spatially AND temporally in one shot — no per-timestep loop
    ds_filtered = ds[var].sel(
        time=slice("2011-01-01", "2015-12-31"),
        lat=slice(LAT_MAX, LAT_MIN),
        lon=slice(LON_MIN, LON_MAX),
    )

    df = ds_filtered.to_dataframe().reset_index()
    df = df.dropna(subset=[var])
    df['time'] = df['time'].astype(str)
    df = df.rename(columns={"lat": "Latitude", "lon": "Longitude", "time": "Sample Date"})

    logging.info(f"Filtering for {var} completed: {df.shape}")
    return df


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
        sample_date = sa_df.loc[i, 'Sample Date']
        nearest_lat = sa_df.loc[i, 'nearest_lat']
        nearest_lon = sa_df.loc[i, 'nearest_lon']

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


# ── Extract all variables — reload dataset each time for fresh SAS token ──────

def extract_all_variables(df, label="dataset"):
    logging.info(f"Extracting {len(VARIABLES)} TerraClimate variables for {label}...")
    results = df[['Latitude', 'Longitude', 'Sample Date']].copy()

    for var in VARIABLES:
        try:
            # Fresh dataset load per variable = fresh SAS token = no auth expiry
            ds = load_terraclimate_dataset()

            if var not in ds.data_vars:
                logging.warning(f"Variable {var} not found in dataset, skipping.")
                results[var] = np.nan
                continue

            climate_df     = filterg(ds, var)
            climate_values = assign_nearest_climate(df.copy(), climate_df, var)
            results[var]   = climate_values
            logging.info(f"✅ {var} extracted successfully")

        except Exception as e:
            logging.error(f"❌ Failed to extract {var}: {e}")
            results[var] = np.nan

    return results


# ── Derived features ──────────────────────────────────────────────────────────

def add_derived_features(df):
    logging.info("Computing derived climate features...")

    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['temp_range'] = df['tmax'] - df['tmin']

    if 'ppt' in df.columns and 'pet' in df.columns:
        df['aridity_index'] = df['ppt'] / (df['pet'] + 1e-10)

    if 'aet' in df.columns and 'pet' in df.columns:
        df['evap_fraction'] = df['aet'] / (df['pet'] + 1e-10)

    if 'ppt' in df.columns and 'aet' in df.columns:
        df['water_balance'] = df['ppt'] - df['aet']

    logging.info(f"Derived features added. Final shape: {df.shape}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Training data
    Water_Quality_df = pd.read_csv('water_quality_training_dataset.csv')
    logging.info(f"Training dataset: {Water_Quality_df.shape}")

    train_results = extract_all_variables(Water_Quality_df, label="training")
    train_results = add_derived_features(train_results)
    train_results.to_csv('terraclimate_features_training.csv', index=False)
    logging.info(f"✅ Training extraction complete: {train_results.shape}")
    print(train_results.isnull().mean().round(2))

    # Validation data
    val_file = 'submission_template.csv' if os.path.exists('submission_template.csv') else 'submission.csv'
    logging.info(f"Processing validation dataset from {val_file}...")
    Validation_df = pd.read_csv(val_file)
    logging.info(f"Validation dataset: {Validation_df.shape}")

    val_results = extract_all_variables(Validation_df, label="validation")
    val_results = add_derived_features(val_results)
    val_results.to_csv('terraclimate_features_validation.csv', index=False)
    logging.info(f"✅ Validation extraction complete: {val_results.shape}")
    print(val_results.isnull().mean().round(2))