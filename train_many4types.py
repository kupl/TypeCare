from pathlib import Path
import ast
import os
from pprint import pprint
from analysis.naive_static_analysis import analyze_code, ProjectUsageInfos
from analysis.typet5 import count_annotations
from run.make_data import make_data
from typet5.type_check import parse_type_str
import glob
from collections import Counter, defaultdict
from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro_winkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import multiprocessing
import pickle
import pandas as pd
import random

IS_RETURN = False  # Set to True if you want to process return types only

TEST_FILE = []
with open(Path.home() / "ManyTypes4PyDataset-v0.7" / "dataset_split.csv", 'r') as f:
    lines = f.readlines()

train_repo_set = set([])
test_repo_set = set([])

for line in lines:
    content = line.split(',')
    repo_path = str(Path.home() / "TypeGen" / "repos" / content[1].split('/')[1])
    if content[0] == "test":
        TEST_FILE.append('/'.join(content[1].split('/')[1:]).strip())


def normalize_type_str(type_str):
    return str(parse_type_str(type_str).normalized())

def levenstein_similarity(str1, str2):
    return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2))

def smart_levenshtein(a, b):
    lev_dist = levenshtein_distance(a, b)
    len_penalty = abs(len(a) - len(b)) / max(len(a), len(b))
    base_score = 1 - lev_dist / max(len(a), len(b))
    return base_score * (1 - len_penalty)

def jaro_winkler_similarity(a, b):
    return jaro_winkler(a, b)

def reversed_jaro_winkler_similarity(a, b):
    return jaro_winkler(a[::-1], b[::-1])

# def cos_similarity(str1, str2):
#     vectorizer = TfidfVectorizer()
#     str1 = str1.replace("_", " ")
#     str2 = str2.replace("_", " ")
#     # assert str1 != "" and str2 != "", f"str1: {str1}, str2: {str2}"
#     try:
#         vectors = vectorizer.fit_transform([str1, str2])
#     except:
#         return -1
#     if cosine_similarity(vectors[0], vectors[1])[0][0] > 0:
#         print(vectors)
#         print(cosine_similarity(vectors[0], vectors[1])[0][0])
#         input()
#     else:
#         print(vectors)
#         print(str1, str2)
#         print(cosine_similarity(vectors[0], vectors[1]))
#         input()
#     return cosine_similarity(vectors[0], vectors[1])[0][0]

def difflib_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def train(train_datas, is_return=False):
    X_train = train_datas.iloc[:, :-1].values
    y_train = train_datas.iloc[:, -1].values

    retrain_model = True

    name = "many_random_forest_model_all.pkl" if not is_return else "many_ret_random_forest_model_all.pkl"

    if not retrain_model and os.path.isfile(name):
        with open(name, 'rb') as f:
            clf = pickle.load(f)
        print("Model loaded from file.")
    else:
        # Train a Random Forest classifier
        clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=16, max_depth=10)

        # print(X_train)
        # print(y_train)
        print("Training Random Forest classifier...")
        clf.fit(X_train, y_train)
        print("Training completed.")

    tree_depths = [estimator.tree_.max_depth for estimator in clf.estimators_]
    avg_depth = sum(tree_depths) / len(tree_depths)
    print(f"평균 트리 깊이: {avg_depth:.2f}")
    # # Make predictions on the validation set
    # y_pred = clf.predict(X_validate)
    # # Calculate accuracy
    # accuracy = accuracy_score(y_validate, y_pred)
    # Save the model
    with open(name, 'wb') as f:
        pickle.dump(clf, f)

    # print(f"Accuracy: {accuracy:.2f}")
    # print(classification_report(y_validate, y_pred, target_names=["0", "1"]))
    # print(f"Feature importances: {clf.feature_importances_}")

def get_test_accuracy(test_datas, is_return=False):
    X_test = test_datas.iloc[:, :-1].values
    y_test = test_datas.iloc[:, -1].values

    name = "many_random_forest_model_all.pkl" if not is_return else "many_ret_random_forest_model_all.pkl"

    with open(name, 'rb') as f:
        clf = pickle.load(f)
    print("Model loaded from file.")

    # Make predictions on the validation set
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=["0", "1"]))
    print(f"Feature importances: {clf.feature_importances_}")

def process_project(project):
    # iter projects and collect data
    args_datas = []
    ret_datas = []

    proj_usage_infos = ProjectUsageInfos()
    proj_usage_infos.update_usage_infos(project)
    annotations = {}
    all_functions = {}

    # iter python files
    # files = glob.glob(f"{project}/**/*.py", recursive=True)
    # files += glob.glob(f"{project}/*.pyi", recursive=True)

    start_time = time.time()
    repo_folder = '/'.join(str(project).split('/')[-1].split('__'))
    for file in glob.glob(f"{project}/**/*.py", recursive=True):
        file_path = repo_folder + '/' + '/'.join(file.split('/')[8:])
        if file_path in TEST_FILE:
            # print(f"Skipping test file: {file_path}")
            continue
        # print(f"Processing file: {project}")
        with open(file, 'r') as f:
            code = f.read()
        functions = analyze_code(code)
        annotation_info = count_annotations(code)

        annotations[file] = annotation_info
        all_functions[file] = functions

    end_time = time.time()
    # print(f"Time taken to analyze {project}: {end_time - start_time:.2f} seconds")

    param_counter_list = []
    ret_counter_list = []

    for file, _ in all_functions.items():
        assert file in annotations, f"File {file} not found in annotations, {annotations.keys()}"
        param_count, ret_count = annotations[file]

        param_counter_list.append(param_count)
        ret_counter_list.append(ret_count)
    
    start_time = time.time()

    for file, classes in all_functions.items():
        for class_name, functions in classes.items():
            for func_name, func_info in functions.items():
                full_name = f"{class_name}.{func_name}" if class_name else func_name

                param_counter = defaultdict(Counter)
                ret_counter = defaultdict(Counter)

                for annot_list in param_counter_list:
                    for func, annots in annot_list.items():
                        if func == full_name:
                            continue
                        for annot in annots:
                            for var, typ in annot.items():
                                param_counter[var][typ] += 1

                for annot_list in ret_counter_list:
                    for func, annots in annot_list.items():
                        if func == full_name:
                            continue
                        for annot in annots:
                            for var, typ in annot.items():
                                ret_counter[var][typ] += 1

                ctx_types = proj_usage_infos.extract_ctx_types(project, file, class_name, func_name)
                target_ctx_types = proj_usage_infos.extract_target_ctx_types(project, file, class_name, func_name)

                # print(f"Project: {project}")
                # print(f"Class: {class_name}")
                # print(f"Function: {func_name}")
                # print(f"Param Counter: {param_counter}")
                # print(f"Ret Counter: {ret_counter}")
                # print(f"Ctx Types: {ctx_types}")
                # print(f"Target Ctx Types: {target_ctx_types}")
                # print(f"Func Info: {func_info}")

                args = func_info["Args"]
                ret = func_info["Ret"]

                # get oracle datas
                arg_oracle_datas = []
                ret_oracle_datas = []

                for arg, value in args.items():
                    if value.get('annotation', None) is None:
                        continue

                    try:

                        oracle_type = normalize_type_str(value['annotation'])
                        cur_ctx = target_ctx_types.param_ctx_types[arg]

                        arg_oracle_datas.append([arg, oracle_type, cur_ctx])
                    except SyntaxError as e:
                        # print(f"Error parsing annotation for {arg}: {e}")
                        continue

                if ret.get('annotation', None) is not None:
                    try:
                        ret_type = normalize_type_str(ret['annotation'])
                        cur_ctx = target_ctx_types.ret_ctx_types[func_name]
                        ret_oracle_datas.append([func_name, ret_type, cur_ctx])
                    except SyntaxError as e:
                        # print(f"Error parsing annotation for {func_name}: {e}")
                        continue
                
                # get diff datas
                args_diff_datas = []
                ret_diff_datas = []

                for name, oracle_type, name_infos in arg_oracle_datas:
                    for name_info, cur_ctx in name_infos.items():
                        (filename, class_hierarchy, func_name, args) = name_info
                        for candidate_name, candidate_name_infos in ctx_types.param_ctx_types.items():
                            for candidate_name_info, candidate_ctx in candidate_name_infos.items():
                                (candidate_filename, candidate_classs_hierarchy, candidate_func_name, candidate_args) = candidate_name_info
                                decorators = set(target_ctx_types.decorators)
                                candidate_decorators = set(ctx_types.decorators)

                                datas = make_data(
                                    cur_ctx, filename, class_hierarchy, func_name, decorators, args, name,
                                    candidate_ctx, candidate_filename, candidate_classs_hierarchy, candidate_func_name, candidate_decorators, candidate_args, candidate_name,
                                    is_constraint=False
                                )

                                # change last data
                                for i in range(len(datas)):
                                    datas[i][-1] = int(oracle_type in datas[i][-1])
                                
                                args_diff_datas.extend(datas)

                for name, oracle_type, name_infos in ret_oracle_datas:
                    for name_info, cur_ctx in name_infos.items():
                        (filename, class_hierarchy, func_name, args) = name_info
                        for candidate_name, candidate_name_infos in ctx_types.ret_ctx_types.items():
                            for candidate_name_info, candidate_ctx in candidate_name_infos.items():
                                (candidate_filename, candidate_classs_hierarchy, candidate_func_name, candidate_args) = candidate_name_info
                                decorators = set(target_ctx_types.decorators)
                                candidate_decorators = set(ctx_types.decorators)

                                datas = make_data(
                                    cur_ctx, filename, class_hierarchy, func_name, decorators, args, name,
                                    candidate_ctx, candidate_filename, candidate_classs_hierarchy, candidate_func_name, candidate_decorators, candidate_args, candidate_name,
                                    is_return=True, is_constraint=False
                                )

                                # change last data
                                for i in range(len(datas)):
                                    datas[i][-1] = int(oracle_type in datas[i][-1])
                            

                                ret_diff_datas.extend(datas)

                args_datas.extend(args_diff_datas)
                ret_datas.extend(ret_diff_datas)

    return args_datas, ret_datas

def run():
    proj_usage_infos = ProjectUsageInfos()
    annotation_infos = {}
    recreate_data = True
    
    is_only_args = False
    is_only_ret = False

    cpu_count = min(multiprocessing.cpu_count(), 16)

    train_repo_path = Path.home() / "TypeT5" / "ManyTypes4Py" / "repos" / "train"
    test_repo_path = Path.home() / "TypeT5" / "ManyTypes4Py" / "repos" / "test"
    validate_repo_path = Path.home() / "TypeT5" / "ManyTypes4Py" / "repos" / "valid"

    # train_projects = list(train_repo_path.iterdir())
    # test_projects = list(test_repo_path.iterdir())
    # validate_projects = list(validate_repo_path.iterdir())
    
    # total_projects = train_projects + test_projects + validate_projects
    
    # repo_path = Path.home() / "ManyTypes4PyDataset-v0.7"
    # # read dataset_split.csv
    # with open(repo_path / "dataset_split.csv", 'r') as f:
    #     lines = f.readlines()

    # train_repo_set = set([])
    # test_repo_set = set([])

    # train_num = 0
    # test_num = 0

    # for line in lines:
    #     content = line.split(',')
    #     repo_path = str(Path.home() / "TypeGen" / "repos" / content[1].split('/')[1])
    #     if content[0] == "train":
    #         train_repo_set.add(content[1].split('/')[1])
    #         train_num += 1
    #     elif content[0] == "test":
    #         test_repo_set.add(content[1].split('/')[1])
    #         test_num += 1

    # print(f"Train Repos: {train_num}")
    # print(f"Test Repos: {test_num}")
    # print(f"Total Projects: {len(total_projects)}")
    # filtered_projects = []

    # for project in total_projects:
    #     is_valid = True
    #     for test_repo in test_repo_set:
    #         if test_repo == project.name.split('__')[0]:
    #             is_valid = False
    #             break

    #     if not is_valid:
    #         continue

    #     filtered_projects.append(project)

    # print(f"Filtered Projects: {len(filtered_projects)}")


    # print(f"Train Repos: {len(train_repo_set)}")
    # print(f"Test Repos: {len(test_repo_set)}")
    
    # train_sample_size = max(1, int(len(train_repo_set) * 0.2))
    # test_sample_size = max(1, int(len(test_repo_set) * 0.1))

    # random.seed(42)
    # sampled_train_repo_list = random.sample(list(train_repo_set), train_sample_size)
    # sampled_test_repo_list = random.sample(list(test_repo_set), test_sample_size)

    train_datas = []
    column_names = [
        "leven_sim", 
        "jaro_sim",
        "reversed_jaro_sim",
        "is_plural",
        "ctx_sim", 
        "ctx_len", 
        "target_ctx_len", 
        "ctx_raio",
        "target_ctx_raio",
        "is_target_global", 
        "is_cand_global", 
        "same_file", 
        "leven_sim_class",
        "jaro_sim_class",
        "reversed_jaro_sim_class",
        "leven_sim_func",
        "jaro_sim_func",
        "reversed_jaro_sim_func",
        "is_include_func",
        "is_included_func",
        "decorator_sim",
        "decorators_len",
        "cand_decorators_len",
        "target_args_len",
        "cand_args_len",
        "same_args", 
        "label"]

    # check file data
    if not recreate_data and os.path.isfile("many_train_data_all.parquet"):
        train_datas = pd.read_parquet('many_train_data_all.parquet', engine='pyarrow')
        ret_train_datas = pd.read_parquet('many_ret_train_data_all.parquet', engine='pyarrow')
    else:
        projects = list(train_repo_path.iterdir())

        with multiprocessing.Pool(processes=cpu_count) as pool:
            results = list(tqdm(pool.imap(process_project, projects), total=len(projects), desc="Processing Projects"))

        train_datas = pd.DataFrame([], columns=column_names)
        ret_train_datas = pd.DataFrame([], columns=column_names)

        args_results = []
        ret_results = []

        for res in results:
            args_results.append(res[0])
            ret_results.append(res[1])

        for res in args_results:
            df_new = pd.DataFrame(res, columns=column_names)
            train_datas = pd.concat([train_datas, df_new], ignore_index=True)

        for res in ret_results:
            df_new = pd.DataFrame(res, columns=column_names)
            # drop 1st~3rd columns
            df_new = df_new.drop(df_new.columns[0:3], axis=1)
            # rename columns
            df_new.columns = column_names[3:]

            ret_train_datas = pd.concat([ret_train_datas, df_new], ignore_index=True)

        train_datas.to_parquet('many_train_data_all.parquet', index=False)
        ret_train_datas.to_parquet('many_ret_train_data_all.parquet', index=False)
        train_datas = pd.read_parquet('many_train_data_all.parquet', engine='pyarrow')
        ret_train_datas = pd.read_parquet('many_ret_train_data_all.parquet', engine='pyarrow')

    print(len(train_datas))
    print(train_datas.head(10))
    print(len(ret_train_datas))
    print(ret_train_datas.head(10))

    # validate_repo_path = Path.home() / "TypeT5" / "ManyTypes4Py" / "repos" / "valid"

    # validate_datas = []
    # if not recreate_data and os.path.isfile("validate_data.parquet"):
    #     validate_datas = pd.read_parquet('validate_data.parquet', engine='pyarrow')
    # else:
    #     projects = list(validate_repo_path.iterdir())

    #     with multiprocessing.Pool(processes=cpu_count) as pool:
    #         results = list(tqdm(pool.imap(process_project, projects), total=len(projects), desc="Processing Projects"))

    #     validate_datas = pd.DataFrame([], columns=column_names)

    #     for res in results:
    #         df_new = pd.DataFrame(res, columns=column_names)
    #         validate_datas = pd.concat([validate_datas, df_new], ignore_index=True)

    #     validate_datas.to_parquet('validate_data.parquet', index=False)

    # print(len(validate_datas))
    # print(validate_datas.head(10))


    test_datas = []
    if not recreate_data and os.path.isfile("test_data_all.parquet"):
        test_datas = pd.read_parquet('test_data_all.parquet', engine='pyarrow')
        ret_test_datas = pd.read_parquet('ret_test_data_all.parquet', engine='pyarrow')
    else:
        projects = list(test_repo_path.iterdir())

        with multiprocessing.Pool(processes=cpu_count) as pool:
            results = list(tqdm(pool.imap(process_project, projects), total=len(projects), desc="Processing Projects"))

        test_datas = pd.DataFrame([], columns=column_names)
        ret_test_datas = pd.DataFrame([], columns=column_names)

        args_results = []
        ret_results = []

        for res in results:
            args_results.append(res[0])
            ret_results.append(res[1])

        for res in args_results:
            df_new = pd.DataFrame(res, columns=column_names)
            test_datas = pd.concat([test_datas, df_new], ignore_index=True)
        
        for res in ret_results:
            df_new = pd.DataFrame(res, columns=column_names)
            # drop 1st~3rd columns
            df_new = df_new.drop(df_new.columns[0:3], axis=1)
            # rename columns
            df_new.columns = column_names[3:]
            ret_test_datas = pd.concat([ret_test_datas, df_new], ignore_index=True)


        test_datas.to_parquet('test_data_all.parquet', index=False)
        ret_test_datas.to_parquet('ret_test_data_all.parquet', index=False)
        test_datas = pd.read_parquet('test_data_all.parquet', engine='pyarrow')
        ret_test_datas = pd.read_parquet('ret_test_data_all.parquet', engine='pyarrow')

    print(len(ret_test_datas))
    print(ret_test_datas.head(10))

    train(train_datas)
    get_test_accuracy(test_datas)

    train(ret_train_datas, is_return=True)
    get_test_accuracy(ret_test_datas, is_return=True)

if __name__ == "__main__":
    run()