import os
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download , snapshot_download

# =========================
# 🔐 AUTHENTICATION
# =========================
HUGGINGFACE_HUB_TOKEN  = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not HUGGINGFACE_HUB_TOKEN:
    raise ValueError("Please set HUGGINGFACE_HUB_TOKEN as environment variable")

login(token=HUGGINGFACE_HUB_TOKEN)

# =========================
# 📂 CONFIG
# =========================
REPO_ID = "gaia-benchmark/GAIA"   # dataset repo
BASE_DIR = "./gaia_data"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# 📥 LOAD DATASET
# =========================
print("Loading GAIA dataset...")

#dataset = load_dataset('gaia-benchmark/GAIA', '2023_level1')
with open(os.path.join(BASE_DIR, "level1_tasks.txt"), "w", encoding="utf-8") as f:
    data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
    dataset = load_dataset(data_dir, "2023_level1", split="test")
    for example in dataset:
        task_id = example["task_id"]
        question = example["Question"]
        file_path = os.path.join(data_dir, example["file_path"])
        f.write(f"Task ID: {task_id} ; Question: {question} ; File Path: {file_path}\n")
f.close()



# print(f"Total Level 1 tasks: {len(level1_tasks)}")
# with open(os.path.join(BASE_DIR, "level1_tasks.txt"), "w", encoding="utf-8") as f:
#     for i in dataset:
#         f.write(str(i))
#         #f.write(f"Task ID: {i['task_id']} ;")
#         #f.write(f"Question: {i['question']} ;")
#         #f.write(f"File Name: {i.get('file_name', 'None')};")
#         #f.write("-" * 50 + ";\n")

# =========================
# 📥 FILE DOWNLOAD FUNCTION
# =========================
# def download_file(file_name):
#     try:
#         local_path = hf_hub_download(
#             repo_id=REPO_ID,
#             filename=f"files/{file_name}",
#             repo_type="dataset",
#             local_dir=BASE_DIR,
#             local_dir_use_symlinks=False
#         )
#         return local_path
#     except Exception as e:
#         print(f"❌ Failed to download {file_name}: {e}")
#         return None

# # =========================
# # 📖 LOAD FILE CONTENT
# # =========================
# def load_file_content(path):
#     if not path:
#         return None
    
#     try:
#         if path.endswith((".py", ".txt", ".md", ".json")):
#             with open(path, "r", encoding="utf-8") as f:
#                 return f.read()
#         else:
#             return f"[Binary or unsupported file: {path}]"
#     except Exception as e:
#         return f"[Error reading file: {e}]"

# # =========================
# # 🔄 PROCESS TASKS
# # =========================
# tasks_with_files = []

# for task in level1_tasks:
#     file_path = None
#     file_content = None

#     if task.get("file_name"):
#         file_path = download_file(task["file_name"])
#         file_content = load_file_content(file_path)

#     tasks_with_files.append({
#         "task_id": task["task_id"],
#         "question": task["question"],
#         "file_name": task.get("file_name"),
#         "file_path": file_path,
#         "file_content_preview": file_content[:500] if file_content else None
#     })

# # =========================
# # 🧪 SAMPLE OUTPUT
# # =========================
# print("\n=== SAMPLE TASK ===\n")
# for t in tasks_with_files[:2]:
#     print("Task ID:", t["task_id"])
#     print("Question:", t["question"])
#     print("File:", t["file_name"])
#     print("Local Path:", t["file_path"])
#     print("Preview:", t["file_content_preview"])
#     print("-" * 50)
