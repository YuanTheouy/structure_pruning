# verify_piqa_dataset.py
import datasets
import logging
import os

# 确保Hugging Face的并行处理功能被禁用，这在某些环境中可以避免挂起
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置日志，方便观察过程
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_piqa():
    try:
        logger.info("Attempting to download and load the 'piqa' dataset...")
        # 这将触发一次干净的下载，因为它在缓存中找不到文件
        piqa_dataset = datasets.load_dataset('piqa')
        
        logger.info("="*50)
        logger.info("🎉 SUCCESS! The 'piqa' dataset was downloaded and loaded correctly.")
        logger.info("Dataset information:")
        print(piqa_dataset)
        logger.info("="*50)
        logger.info("You can now proceed to run your main script with all 7 tasks.")

    except Exception as e:
        logger.error("="*50)
        logger.error(f"🔥🔥🔥 FAILURE: Could not load the 'piqa' dataset.")
        logger.error(f"Error details: {e}")
        logger.error("Please check your network connection or for potential Hugging Face Hub issues.")
        logger.error("="*50)

if __name__ == "__main__":
    # 在执行主要逻辑前，我们先执行上一轮的清理步骤，确保万无一失
    downloads_cache_dir = os.path.expanduser('~/.cache/huggingface/datasets/downloads')
    if os.path.exists(downloads_cache_dir):
        logger.info(f"Clearing the downloads cache at: {downloads_cache_dir}")
        # 安全起见，我们只删除文件，保留目录本身
        for filename in os.listdir(downloads_cache_dir):
            file_path = os.path.join(downloads_cache_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f'Failed to delete {file_path}. Reason: {e}')
        logger.info("Downloads cache cleared.")
    
    verify_piqa()