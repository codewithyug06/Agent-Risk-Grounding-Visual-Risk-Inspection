import os
import subprocess

# 1. Force Hugging Face to use D: Drive for all temporary caching
# This MUST be set before importing load_dataset
os.environ["HF_HOME"] = r"D:\Computer Vision Project\HF_Cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\Computer Vision Project\HF_Cache\datasets"

from datasets import load_dataset

# 2. Define directories
BASE_DIR = r"D:\Computer Vision Project\All_Datasets"
CACHE_DIR = r"D:\Computer Vision Project\HF_Cache\datasets"

# 3. List of datasets
datasets_to_download = [
    {"repo_id": "osunlp/Multimodal-Mind2Web", "folder_name": "Multimodal-Mind2Web"},
    {"repo_id": "osunlp/Mind2Web", "folder_name": "Mind2Web"},
    {"repo_id": "xlangai/AgentTrek", "folder_name": "AgentTrek"},
    {"repo_id": "bevaya/ScreenSpot", "folder_name": "ScreenSpot"},
    {"repo_id": "lmms-lab/ScreenSpot-v2", "folder_name": "ScreenSpot-v2"}
]

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("Starting mass download process using D: drive for caching...")

for ds_info in datasets_to_download:
    repo_id = ds_info["repo_id"]
    save_path = os.path.join(BASE_DIR, ds_info["folder_name"])
    
    print(f"==================================================")
    print(f"Processing: {repo_id}")
    
    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"⏭️ Skipping {repo_id} - Folder already exists at {save_path}")
        continue

    try:
        print(f"⏳ Downloading and caching {repo_id} to D: drive...")
        # Point the cache directly to the D: drive directory
        dataset = load_dataset(repo_id, cache_dir=CACHE_DIR)
        
        print(f"💾 Saving to separate folder: {save_path}...")
        dataset.save_to_disk(save_path)
        print(f"✅ Successfully saved {repo_id}!\n")
        
    except Exception as e:
        print(f"❌ Error downloading {repo_id}. Error details: {e}")
        print("Moving to the next dataset...\n")

print("==================================================")
print("Processing: VisualWebArena GitHub Repository")
git_folder = os.path.join(BASE_DIR, "visualwebarena")

if os.path.exists(git_folder):
    print(f"⏭️ Skipping git clone - Folder already exists at {git_folder}")
else:
    print(f"⏳ Cloning visualwebarena into {git_folder}...")
    try:
        subprocess.run(["git", "clone", "https://github.com/web-arena-x/visualwebarena", git_folder], check=True)
        print("✅ Successfully cloned visualwebarena!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository. Error: {e}")

print("==================================================")
print(f"🎉 All tasks complete! Final files are in: {BASE_DIR}")