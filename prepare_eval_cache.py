import evaluate
import datasets
from transformers import logging as transformers_logging
import logging
import os
import shutil

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()

# Assets to cache
METRICS_TO_CACHE = ["exact_match", "rouge", "bleu", "sacrebleu"]
DATASETS_TO_CACHE = [
    {"name": "boolq", "subset": None, "split": "validation"},
    {"name": "piqa", "subset": None, "split": "validation"},
    {"name": "hellaswag", "subset": "default", "split": "test"},
    {"name": "winogrande", "subset": "winogrande_debiased", "split": "validation"},
    {"name": "ai2_arc", "subset": "ARC-Easy", "split": "test"},
    {"name": "ai2_arc", "subset": "ARC-Challenge", "split": "test"},
    {"name": "openbookqa", "subset": "main", "split": "test"},
]
HF_CACHE_HOME = os.path.expanduser(os.path.join('~', '.cache', 'huggingface'))
# --- End of Configuration ---

def clear_dataset_cache(dataset_name, subset=None):
    """A helper function to clear cache for a specific dataset."""
    try:
        # Construct the dataset path within the cache directory
        if dataset_name == 'piqa':
             # PIQA has a specific, non-standard path
             path_name = 'piqa'
        else:
            path_name = dataset_name if subset is None or subset == 'default' else os.path.join(dataset_name, subset)

        dataset_cache_dir = os.path.join(HF_CACHE_HOME, 'datasets', path_name)

        if os.path.exists(dataset_cache_dir):
            logger.warning(f"Found and deleting corrupted cache for '{path_name}': {dataset_cache_dir}")
            shutil.rmtree(dataset_cache_dir)
            logger.info("Cache cleared successfully.")
            return True
        else:
            logger.warning(f"Cache directory not found for '{path_name}', nothing to clear.")
            return True # If it doesn't exist, we can proceed
    except Exception as e:
        logger.error(f"Error while trying to clear cache for {dataset_name}: {e}")
    return False

def pre_cache_assets():
    """Download and cache all necessary evaluation resources."""
    logger.info("--- 1. Verifying metrics in cache ---")
    all_metrics_ok = True
    for metric in METRICS_TO_CACHE:
        try:
            # === CRITICAL FIX ===
            # Added trust_remote_code=True to allow loading local scripts
            evaluate.load(os.path.join(HF_CACHE_HOME, 'evaluate', 'metrics', metric), trust_remote_code=True)
            logger.info(f"✅ Metric '{metric}' found and loaded successfully from cache.")
        except Exception as e:
            logger.error(f"❌ Failed to load metric '{metric}' from cache. Please ensure manual_cache_metrics.sh ran correctly. Error: {e}")
            all_metrics_ok = False
    
    if not all_metrics_ok:
        logger.critical("One or more metrics failed to load. Aborting further checks.")
        return

    logger.info("\n--- 2. Caching evaluation datasets ---")
    for item in DATASETS_TO_CACHE:
        name, subset, split = item['name'], item['subset'], item['split']
        display_name = f"{name}" + (f" ({subset})" if subset and subset != 'default' else "")
        
        try:
            logger.info(f"Attempting to cache dataset: '{display_name}' [split: {split}]")
            datasets.load_dataset(name, subset, split=split, trust_remote_code=True)
            logger.info(f"✅ Successfully cached '{display_name}'.")
        except Exception as e:
            if 'checksum' in str(e).lower() or isinstance(e, UnicodeDecodeError):
                logger.error(f"❌ Checksum or decode error for '{display_name}'. This suggests a corrupted download.")
                if clear_dataset_cache(name, subset):
                    logger.info(f"Retrying to download '{display_name}' after clearing cache...")
                    try:
                        datasets.load_dataset(name, subset, split=split, trust_remote_code=True)
                        logger.info(f"✅ Successfully cached '{display_name}' on second attempt.")
                    except Exception as retry_e:
                        logger.error(f"❌ Still failed to cache '{display_name}' after retry: {retry_e}")
                else:
                    logger.error(f"Could not clear cache for '{display_name}'.")
            else:
                 logger.error(f"❌ Failed to cache '{display_name}': {e}")


    logger.info("\n--- Caching process finished! ---")

if __name__ == "__main__":
    pre_cache_assets()