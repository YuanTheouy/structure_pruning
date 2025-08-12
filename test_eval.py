import sys
import os

print("--- 诊断信息开始 ---")

# 1. 打印当前正在使用的 Python 解释器
print(f"[*] Python Executable: {sys.executable}")
print("-" * 20)

# 2. 打印模块搜索路径 (sys.path)
# 这会告诉我们 Python 在哪些文件夹里寻找库
print("[*] sys.path:")
for path in sys.path:
    print(f"    - {path}")
print("-" * 20)

# 3. 尝试导入 lm_eval 并打印信息
try:
    import lm_eval
    print("[SUCCESS] 成功导入 'lm_eval' 模块！")
    print(f"    - 版本 (Version): {getattr(lm_eval, '__version__', 'N/A')}")
    # __file__ 属性会告诉我们模块是从哪个文件加载的
    print(f"    - 位置 (Path):    {os.path.abspath(lm_eval.__file__)}")
except ImportError as e:
    print(f"[FAILURE] 导入失败! 未找到 'lm_eval' 模块。")
    print(f"    - 错误信息: {e}")
except Exception as e:
    print(f"[FAILURE] 导入时发生未知错误。")
    print(f"    - 错误类型: {type(e).__name__}")
    print(f"    - 错误信息: {e}")

print("--- 诊断信息结束 ---")
