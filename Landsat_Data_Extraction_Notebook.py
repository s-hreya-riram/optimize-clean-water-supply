import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import logging
import random

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('landsat_extraction.log'),
        logging.StreamHandler()
    ]
)

print("✅ Dependencies loaded successfully!")


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_throttle_error(e):
    """Detect rate-limit / throttling responses from Planetary Computer."""
    msg = str(e).lower()
    return any(k in msg for k in ['429', 'too many requests', 'rate limit', 'throttl'])


def backoff_sleep(attempt, base_delay=2, max_delay=60):
    """
    Exponential backoff with jitter.
    attempt=0 -> ~2s, attempt=1 -> ~4s, attempt=2 -> ~8s, capped at max_delay.
    Jitter prevents all workers retrying at the same instant.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.3)
    sleep_time = delay + jitter
    logging.info(f"Backing off for {sleep_time:.1f}s (attempt {attempt + 1})")
    time.sleep(sleep_time)


# ── Spectral indices ──────────────────────────────────────────────────────────

def compute_spectral_indices(blue, green, red, nir, swir1, swir2):
    eps = 1e-10
    indices = {}

    # Vegetation
    indices['NDVI']  = (nir - red)  / (nir + red  + eps)
    indices['EVI']   = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    indices['SAVI']  = ((nir - red) / (nir + red + 0.5)) * 1.5
    indices['ARVI']  = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + eps)
    indices['GNDVI'] = (nir - green) / (nir + green + eps)
    indices['RDVI']  = (nir - red)  / np.sqrt(nir + red + eps)

    # Water
    indices['NDWI']    = (green - nir)   / (green + nir   + eps)
    indices['MNDWI']   = (green - swir1) / (green + swir1 + eps)
    indices['AWEInsh'] = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    indices['AWEIsh']  = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2

    # Soil / built-up
    indices['BSI']  = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
    indices['NDBI'] = (swir1 - nir) / (swir1 + nir + eps)
    indices['UI']   = (swir2 - nir) / (swir2 + nir + eps)

    # Burn / geological
    indices['NBR']          = (nir - swir2) / (nir + swir2 + eps)
    indices['ClayMinerals'] = swir1 / (swir2 + eps)

    # Water quality
    indices['TurbidityIndex']   = (red / (green + eps)) * (swir1 / (nir + eps))
    indices['ChlorophyllIndex'] = (nir / (red + eps)) - 1
    indices['NIR_Red_Ratio']    = nir / (red + eps)
    indices['NDMI']             = (nir - swir1) / (nir + swir1 + eps)

    return indices


def get_output_columns():
    bands   = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'coastal']
    indices = [
        'NDVI', 'EVI', 'SAVI', 'ARVI', 'GNDVI', 'RDVI',
        'NDWI', 'MNDWI', 'AWEInsh', 'AWEIsh',
        'BSI', 'NDBI', 'UI', 'NBR', 'ClayMinerals',
        'TurbidityIndex', 'ChlorophyllIndex', 'NIR_Red_Ratio', 'NDMI',
    ]
    quality = ['cloud_cover', 'data_quality']
    return bands + indices + quality


# ── Per-location extraction ───────────────────────────────────────────────────

def compute_enhanced_Landsat_values(row, retry_count=4, base_delay=2):
    lat      = row['Latitude']
    lon      = row['Longitude']
    date_str = row['Sample Date']

    try:
        date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        if pd.isna(date):
            date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
    except Exception:
        logging.warning(f"Could not parse date: {date_str}")
        return pd.Series({col: np.nan for col in get_output_columns()})

    bbox_size = 0.00089831
    bbox = [
        lon - bbox_size / 2,
        lat - bbox_size / 2,
        lon + bbox_size / 2,
        lat + bbox_size / 2,
    ]

    for attempt in range(retry_count):
        try:
            # Re-open catalog and re-sign on every attempt -> always fresh SAS token
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
                #request_params={"timeout": 30},
            )

            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime="2011-01-01/2015-12-31",
                query={"eo:cloud_cover": {"lt": 10}},
            )
            items = search.item_collection()

            if not items:
                logging.warning(f"No items found for lat={lat}, lon={lon}")
                return pd.Series({col: np.nan for col in get_output_columns()})

            sample_date_utc = (
                date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
            )
            items = sorted(
                items,
                key=lambda x: abs(
                    pd.to_datetime(x.properties["datetime"]).tz_convert("UTC")
                    - sample_date_utc
                ),
            )

            # Sign fresh on every attempt
            selected_item = pc.sign(items[0])

            bands_of_interest = ["blue", "green", "red", "nir08", "swir16", "swir22"]
            data = stac_load(
                [selected_item], bands=bands_of_interest, bbox=bbox
            ).isel(time=0)

            result = {}
            result['blue']    = float(data["blue"].median(skipna=True).values)   if "blue"   in data else np.nan
            result['green']   = float(data["green"].median(skipna=True).values)  if "green"  in data else np.nan
            result['red']     = float(data["red"].median(skipna=True).values)    if "red"    in data else np.nan
            result['nir']     = float(data["nir08"].median(skipna=True).values)  if "nir08"  in data else np.nan
            result['swir16']  = float(data["swir16"].median(skipna=True).values) if "swir16" in data else np.nan
            result['swir22']  = float(data["swir22"].median(skipna=True).values) if "swir22" in data else np.nan
            result['coastal'] = np.nan  # Landsat 8/9 only - skip extra search

            for key in list(result.keys()):
                if result[key] == 0:
                    result[key] = np.nan

            if not any(np.isnan(result.get(b, np.nan)) for b in ['green', 'nir', 'swir16']):
                indices = compute_spectral_indices(
                    blue=result.get('blue', np.nan),
                    green=result['green'],
                    red=result.get('red', np.nan),
                    nir=result['nir'],
                    swir1=result['swir16'],
                    swir2=result['swir22'],
                )
                result.update(indices)
            else:
                for key in compute_spectral_indices(
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                ):
                    result[key] = np.nan

            result['cloud_cover']  = float(selected_item.properties.get('eo:cloud_cover', -1))
            result['data_quality'] = 'good' if result['cloud_cover'] < 5 else 'fair'

            return pd.Series(result)

        except Exception as e:
            if is_throttle_error(e):
                logging.warning(f"Throttled (attempt {attempt + 1}) for lat={lat}, lon={lon}")
                backoff_sleep(attempt, base_delay=base_delay)
            else:
                logging.warning(f"Attempt {attempt + 1} failed for lat={lat}, lon={lon}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(base_delay)

    logging.error(f"All {retry_count} attempts failed for lat={lat}, lon={lon}")
    return pd.Series({col: np.nan for col in get_output_columns()})


# ── Batch processor ───────────────────────────────────────────────────────────

class LandsatBatchProcessor:
    def __init__(
        self,
        batch_size=200,
        checkpoint_dir="./checkpoints",
        max_workers=30,
        inter_batch_sleep=3,
    ):
        self.batch_size        = batch_size
        self.max_workers       = max_workers
        self.inter_batch_sleep = inter_batch_sleep
        self.checkpoint_dir    = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, batch_num, results_df, processed_indices):
        checkpoint_data = {
            'batch_num':         batch_num,
            'processed_indices': list(processed_indices),
            'timestamp':         datetime.now().isoformat(),
        }
        results_path    = self.checkpoint_dir / f"batch_{batch_num:04d}_results.csv"
        checkpoint_path = self.checkpoint_dir / f"batch_{batch_num:04d}_checkpoint.json"
        results_df.to_csv(results_path, index=False)
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logging.info(f"Checkpoint saved for batch {batch_num}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("*_checkpoint.json"), reverse=True
        )
        if not checkpoint_files:
            logging.info("No existing checkpoints - starting fresh")
            return None, set()

        for checkpoint_file in checkpoint_files:
            try:
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                processed_indices = set(checkpoint_data['processed_indices'])
                all_results = []
                for batch_num in range(checkpoint_data['batch_num'] + 1):
                    rp = self.checkpoint_dir / f"batch_{batch_num:04d}_results.csv"
                    if rp.exists():
                        try:
                            all_results.append(pd.read_csv(rp))
                        except Exception as e:
                            logging.warning(f"Could not load batch {batch_num}: {e}")
                combined = (
                    pd.concat(all_results, ignore_index=True)
                    if all_results
                    else pd.DataFrame()
                )
                logging.info(
                    f"Resumed from {checkpoint_file.name}: {len(combined)} locations done"
                )
                return combined, processed_indices
            except Exception as e:
                logging.warning(f"Could not load {checkpoint_file.name}: {e}")
                continue

        logging.warning("All checkpoints unreadable - starting fresh")
        return None, set()

    # ── Parallel batch execution ──────────────────────────────────────────────

    def _process_batch_parallel(self, batch_df, batch_num):
        batch_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(compute_enhanced_Landsat_values, row): idx
                for idx, row in batch_df.iterrows()
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Batch {batch_num + 1}",
            ):
                idx = futures[future]
                try:
                    result = future.result(timeout=120)
                    result.name = idx
                    batch_results.append(result)
                except Exception as e:
                    logging.warning(f"Location {idx} timed out or failed: {e}")
                    empty      = pd.Series({col: np.nan for col in get_output_columns()})
                    empty.name = idx
                    batch_results.append(empty)
        return batch_results

    # ── Main loop ─────────────────────────────────────────────────────────────

    def process_dataset(self, df, output_path, resume=True):
        start_time = datetime.now()

        existing_results, processed_indices = (None, set())
        if resume:
            existing_results, processed_indices = self.load_checkpoint()

        remaining_df = df[~df.index.isin(processed_indices)].copy()

        if len(remaining_df) == 0:
            logging.info("All locations already processed!")
            if existing_results is not None:
                self._finalize_results(existing_results, df, output_path)
            return existing_results

        logging.info(
            f"Processing {len(remaining_df)} remaining locations "
            f"in batches of {self.batch_size} with {self.max_workers} workers"
        )

        all_results = [existing_results] if existing_results is not None else []

        for batch_start in range(0, len(remaining_df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(remaining_df))
            batch_df  = remaining_df.iloc[batch_start:batch_end].copy()
            batch_num = len(processed_indices) // self.batch_size

            logging.info(f"Batch {batch_num + 1}: locations {batch_start + 1}-{batch_end}")
            batch_start_time = datetime.now()

            batch_results    = self._process_batch_parallel(batch_df, batch_num)
            batch_df_results = pd.DataFrame(batch_results)

            batch_df_results['Latitude']    = batch_df['Latitude'].values
            batch_df_results['Longitude']   = batch_df['Longitude'].values
            batch_df_results['Sample Date'] = batch_df['Sample Date'].values

            self.save_checkpoint(
                batch_num,
                batch_df_results,
                processed_indices | set(batch_df.index),
            )

            all_results.append(batch_df_results)
            processed_indices.update(batch_df.index)

            batch_time = (datetime.now() - batch_start_time).total_seconds()
            lps        = len(batch_df) / batch_time if batch_time > 0 else 0
            remaining  = len(df) - len(processed_indices)
            eta        = str(timedelta(seconds=int(remaining / lps))) if lps > 0 else "unknown"

            logging.info(f"Batch {batch_num + 1} done in {batch_time:.1f}s")
            logging.info(
                f"Progress: {len(processed_indices)}/{len(df)} "
                f"({len(processed_indices) / len(df) * 100:.1f}%)"
            )
            logging.info(f"Speed: {lps:.2f} loc/s  |  ETA: {eta}")

            # Polite pause between batches to reduce rate-limit pressure
            if batch_start + self.batch_size < len(remaining_df):
                logging.info(f"Sleeping {self.inter_batch_sleep}s between batches...")
                time.sleep(self.inter_batch_sleep)

        final_results = pd.concat(all_results, ignore_index=True)
        self._finalize_results(final_results, df, output_path)

        total_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Done in {total_time / 3600:.2f} hours!")
        return final_results

    def _finalize_results(self, results_df, original_df, output_path):
        meta_cols    = ['Latitude', 'Longitude', 'Sample Date']
        feature_cols = [c for c in get_output_columns() if c in results_df.columns]
        results_df   = results_df[meta_cols + feature_cols]
        results_df.to_csv(output_path, index=False)
        logging.info(f"Saved to {output_path}")
        logging.info(
            f"Success rate: "
            f"{results_df['nir'].notna().sum() / len(results_df) * 100:.1f}%"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Training data
    Water_Quality_df = pd.read_csv('water_quality_training_dataset.csv')
    print(f"Training dataset: {Water_Quality_df.shape}")

    processor = LandsatBatchProcessor(
        batch_size=200,
        max_workers=30,          # tune down to 20 if you see sustained throttling
        checkpoint_dir="./checkpoints",
        inter_batch_sleep=3,     # 3s pause between every batch
    )

    results = processor.process_dataset(
        df=Water_Quality_df,
        output_path="landsat_features_training.csv",
        resume=True,             # picks up from existing checkpoints automatically
    )
    print(f"Training extraction done: {results.shape}")

    # Validation data
    Validation_df = pd.read_csv('submission_template.csv')
    print(f"Validation dataset: {Validation_df.shape}")

    val_processor = LandsatBatchProcessor(
        batch_size=200,
        max_workers=30,
        checkpoint_dir="./checkpoints_val",
        inter_batch_sleep=3,
    )

    val_results = val_processor.process_dataset(
        df=Validation_df,
        output_path="landsat_features_validation.csv",
        resume=True,
    )
    print(f"Validation extraction done: {val_results.shape}")