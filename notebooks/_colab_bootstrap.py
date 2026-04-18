"""
Paste this cell at the top of every Colab notebook.
Replace GITHUB_REPO with your actual repo URL after pushing.
"""

GITHUB_REPO = "https://github.com/evanpaul14/prompt-classifier.git"
DRIVE_BASE = "/content/drive/MyDrive/polygence"

BOOTSTRAP = f"""
import subprocess, os, sys

# 1. GPU check
subprocess.run(["nvidia-smi"], check=False)

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '{DRIVE_BASE}'

# 3. Clone or pull repo
repo_dir = '/content/polygence'
if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", "{GITHUB_REPO}", repo_dir], check=True)
else:
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

os.chdir(repo_dir)
sys.path.insert(0, 'src')

# 4. Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# 5. Point HF caches at Drive
os.environ['HF_DATASETS_CACHE'] = f'{{DRIVE}}/hf_cache/datasets'
os.environ['HF_HOME']           = f'{{DRIVE}}/hf_cache/hub'

# 6. Symlink Drive dirs so relative paths in code work
for subdir in ['data/processed', 'data/interim', 'models', 'reports']:
    drive_path = f'{{DRIVE}}/{{subdir}}'
    os.makedirs(drive_path, exist_ok=True)
    local_path = subdir
    # Remove existing symlink/dir if needed
    if os.path.islink(local_path):
        os.remove(local_path)
    elif os.path.isdir(local_path) and not os.listdir(local_path):
        os.rmdir(local_path)
    if not os.path.exists(local_path):
        os.symlink(drive_path, local_path)

# 7. Seeds
from prompt_classifier.seeds import set_all_seeds
set_all_seeds(42)
print("Bootstrap complete. CWD:", os.getcwd())
"""

print("Copy the BOOTSTRAP string content into a Colab cell and run exec(BOOTSTRAP)")
