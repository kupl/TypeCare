import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import libcst as cst
from tqdm import tqdm

from typet5.data import GitRepo, get_dataset_dir
from typet5.type_env import collect_annots_info, mypy_checker
from typet5.utils import proj_root, read_file, write_file, not_none, pickle_load

import pickle

BENCHMARK_SCRIPTS_DIR = Path("benchmarks/bettertypes4py/")
repos_dir = Path("BetterTypes4Py/repos")

def count_repo_annots(rep):
    try:
        rep.collect_annotations(repos_dir)
        if rep.n_type_annots / rep.lines_of_code > 0.05:
            return rep
    except Exception as e:
        logging.warning(f"Failed to count annotations for {rep.name}. Exception: {e}")
        return None

def run():
    all_repos = json.loads(read_file(BENCHMARK_SCRIPTS_DIR / "mypy-dependents-by-stars.json"))
    all_repos = [GitRepo.from_json(r) for r in all_repos]

    def download_repos(
        to_download: list[GitRepo], repos_dir, download_timeout=10.0, max_workers=10
    ) -> list[GitRepo]:
        def download_single(repo: GitRepo):
            try:
                if repo.download(repos_dir, timeout=download_timeout):
                    if not repo.repo_dir(repos_dir).exists():
                        subprocess.run(
                            ["mv", Path("downloading") / repo.authorname(), "downloaded"],
                            cwd=repos_dir,
                            capture_output=True,
                        )
                    repo.read_last_update(repos_dir)
                    return repo
                else:
                    return None
            except subprocess.TimeoutExpired:
                return None
            except Exception as e:
                logging.warning(f"Failed to download {repo.name}. Exception: {e}")
                return None

        print("Downloading repos from Github...")
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fs = [executor.submit(download_single, repo) for repo in to_download]
            rs = [f.result() for f in tqdm(as_completed(fs), total=len(fs))]
        print(f"Downloading took {time.time() - t_start} seconds.")
        downloaded = [r for r in rs if r is not None]
        return downloaded

    if not repos_dir.exists():
        (repos_dir / "downloading").mkdir(parents=True)
        (repos_dir / "downloaded").mkdir(parents=True)
        downloaded_repos = download_repos(all_repos, repos_dir)
        print("Deleting failed repos...")
        shutil.rmtree(repos_dir / "downloading")

        print(f"Downloaded {len(downloaded_repos)}/{len(all_repos)} repos.")

    useful_repos = BENCHMARK_SCRIPTS_DIR / "useful_repos.pkl"
    repos_split = pickle_load(BENCHMARK_SCRIPTS_DIR / "repos_split.pkl")

    with open(useful_repos, "rb") as f:
        useful_repos = pickle.load(f)

    print(f"Found {len(useful_repos)} useful repos from previous run.")

    for split, repos in repos_split.items():
        for r in tqdm(repos, desc=f"Moving {split} repos."):
            r: GitRepo
            split: str
            src = repos_dir / "downloaded" / r.authorname()
            (repos_dir / split).mkdir(parents=True, exist_ok=True)
            dest = repos_dir / split / r.authorname()
            if src.exists():
                shutil.move(src, dest)
            else:
                print(f"Repo {r.name} not found.")


if __name__ == "__main__":
    run()