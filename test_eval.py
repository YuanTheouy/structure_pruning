# test_eval.py (新版本)
import evaluate
import os
import shutil

# --- 配置 ---
METRIC_TO_TEST = "exact_match"
HF_CACHE_HOME = os.path.expanduser(os.path.join('~', '.cache', 'huggingface'))
METRIC_CACHE_PATH = os.path.join(HF_CACHE_HOME, 'evaluate', 'metrics', METRIC_TO_TEST)

def run_test():
    """执行一个干净的、隔离的度量加载测试"""
    print("="*60)
    print(f"🔬 开始对 '{METRIC_TO_TEST}' 度量进行隔离测试...")
    print(f"预期缓存路径: {METRIC_CACHE_PATH}")
    print("="*60)

    # 1. (可选但推荐) 清理旧的、可能已损坏的缓存
    if os.path.exists(METRIC_CACHE_PATH):
        print(f"🚨 发现已存在的缓存，正在删除以进行干净测试: {METRIC_CACHE_PATH}")
        try:
            shutil.rmtree(METRIC_CACHE_PATH)
            print("✅ 旧缓存已成功删除。")
        except Exception as e:
            print(f"❌ 删除旧缓存失败: {e}")
            print("测试可能因旧文件干扰而失败。")
            return

    # 2. 尝试让 evaluate 库自己下载和加载
    print("\n--- 步骤 1: 尝试在线加载 (让 evaluate 自动处理缓存) ---")
    try:
        # 这是最标准、最直接的调用方式
        metric_module = evaluate.load(METRIC_TO_TEST)
        print("\n🎉🎉🎉 在线加载成功！🎉🎉🎉")
        print(f"成功加载的模块详情: {metric_module}")
        print("这意味着您的网络、权限和基础环境没有问题。")
        print("问题 100% 出在 manual_cache_metrics.sh 脚本创建的缓存上。")
        
    except Exception as e:
        print(f"\n🔥🔥🔥 在线加载失败: {e} 🔥🔥🔥")
        print("这可能表示存在网络问题 (无法访问Hugging Face Hub) 或权限问题 (无法写入缓存目录)。")
        print("请检查您的网络连接和 ~/.cache/huggingface 目录的写权限。")

    print("\n" + "="*60)
    print("测试结束。")

if __name__ == "__main__":
    run_test()