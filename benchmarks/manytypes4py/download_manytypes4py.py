import git
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.parse import urlparse

def download_repo():
    with open("benchmarks/manytypes4py/ManyTypes4PyDataset.spec", "r") as f:
        spec = f.readlines()

    repos_dir = Path("ManyTypes4Py/repos")

    if not repos_dir.exists():
        repos_dir.mkdir(parents=True)

    def download_single(line):
        splited_line = line.split()
        github_url = splited_line[0]
        if len(splited_line) == 2:
            commit_id = splited_line[1]
        else:
            commit_id = None

        path = urlparse(github_url).path
        repo = path.lstrip('/').removesuffix('.git')

        # print("Downloading", repo)

        repo_dir = repos_dir / repo

        if repo_dir.exists():
            print(f"{repo} already exists, skipping")
            return None

        try:
            repo = git.Repo.clone_from(github_url, repo_dir)
            repo.git.checkout(commit_id)
        except git.exc.GitCommandError:
            # print(f"Failed to download {repo}, skipping")
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            return None
        
        return repo_dir

        # print(f"Checked out commit {commit_id} in {repo_dir}")

    print("Downloading repos from Github...")

    max_workers=10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fs = [executor.submit(download_single, line) for line in spec if "JojiKoike/OMWebAppEngine" in line]
        rs = [f.result() for f in tqdm(as_completed(fs), total=len(fs))]

    print("All repos downloaded.")

if __name__ == "__main__":
    download_repo()