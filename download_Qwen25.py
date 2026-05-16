from huggingface_hub import snapshot_download

# 指定模型ID和本地存储路径
model_id = "Qwen/Qwen2.5-7B"
local_model_path = f"/home/theo/data/yx_repository/01_Models/{model_id.replace('/', '_')}" # 例如, ./Qwen_Qwen1.5-7B

print(f"开始将模型 {model_id} 下载到 {local_model_path}")
snapshot_download(
    repo_id=model_id,
    local_dir=local_model_path,
    local_dir_use_symlinks=False, # 建议设为False，直接下载文件而非创建符号链接
    resume_download=True # 支持断点续传
)
print(f"模型 {model_id} 已完整下载到 {local_model_path}")
