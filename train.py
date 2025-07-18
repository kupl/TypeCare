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

IS_CONSTRAINT = False

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

    print(y_train)

    retrain_model = False

    name = "random_forest_model_all.pkl" if not is_return else "ret_random_forest_model_all.pkl"

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

        with open(name, 'wb') as f:
            pickle.dump(clf, f)

    tree_depths = [estimator.tree_.max_depth for estimator in clf.estimators_]
    avg_depth = sum(tree_depths) / len(tree_depths)
    print(f"평균 트리 깊이: {avg_depth:.2f}")
    # # Make predictions on the validation set
    # y_pred = clf.predict(X_validate)
    # # Calculate accuracy
    # accuracy = accuracy_score(y_validate, y_pred)
    # Save the model

    # print(f"Accuracy: {accuracy:.2f}")
    # print(classification_report(y_validate, y_pred, target_names=["0", "1"]))
    # print(f"Feature importances: {clf.feature_importances_}")

def get_test_accuracy(test_datas, is_return=False):
    X_test = test_datas.iloc[:, :-1].values
    y_test = test_datas.iloc[:, -1].values

    name = "random_forest_model_all.pkl" if not is_return else "ret_random_forest_model_all.pkl"

    with open(name, 'rb') as f:
        clf = pickle.load(f)
    print("Model loaded from file.")

    # Make predictions on the validation set
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=["0", "1"]))

    # Extract top features
    feature_importances = clf.feature_importances_
    feature_names = test_datas.columns[:-1].tolist()
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Top features:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")

    # print(f"Feature importances: {clf.feature_importances_}")

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

    for file in glob.glob(f"{project}/**/*.py", recursive=True):
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
                        print(f"Error parsing annotation for {arg}: {e}")
                        continue

                if ret.get('annotation', None) is not None:
                    try:
                        ret_type = normalize_type_str(ret['annotation'])
                        cur_ctx = target_ctx_types.ret_ctx_types[func_name]
                        ret_oracle_datas.append([func_name, ret_type, cur_ctx])
                    except SyntaxError as e:
                        print(f"Error parsing annotation for {func_name}: {e}")
                        continue
                
                # get diff datas
                args_diff_datas = []
                ret_diff_datas = []

                if not IS_CONSTRAINT:
                    arg_oracle_datas = random.sample(arg_oracle_datas, int(len(arg_oracle_datas) * 0.5))
                    # ret_oracle_datas = random.sample(ret_oracle_datas, int(len(ret_oracle_datas) * 0.5))

                # for name, oracle_type, name_infos in arg_oracle_datas:
                #     for name_info, cur_ctx in name_infos.items():
                #         (filename, class_hierarchy, func_name, args) = name_info
                #         for candidate_name, candidate_name_infos in ctx_types.param_ctx_types.items():
                #             for candidate_name_info, candidate_ctx in candidate_name_infos.items():
                #                 (candidate_filename, candidate_classs_hierarchy, candidate_func_name, candidate_args) = candidate_name_info
                #                 decorators = set(target_ctx_types.decorators)
                #                 candidate_decorators = set(ctx_types.decorators)

                #                 datas = make_data(
                #                     cur_ctx, filename, class_hierarchy, func_name, decorators, args, name,
                #                     candidate_ctx, candidate_filename, candidate_classs_hierarchy, candidate_func_name, candidate_decorators, candidate_args, candidate_name,
                #                     is_constraint=IS_CONSTRAINT
                #                 )

                #                 # change last data
                #                 for i in range(len(datas)):
                #                     datas[i][-1] = int(oracle_type in datas[i][-1])
                                
                #                 args_diff_datas.extend(datas)

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
                                    is_return=True, is_constraint=IS_CONSTRAINT, oracle_type=oracle_type
                                )

                                # change last data
                                for i in range(len(datas)):
                                    datas[i][-1] = int(oracle_type in datas[i][-1])
                            

                                ret_diff_datas.extend(datas)

                # dir_path = Path(str(file)) / full_name
                # if not dir_path.exists():
                #     dir_path.mkdir(parents=True, exist_ok=True)

                # with open(dir_path / "args_datas.pkl", 'wb') as f:


                args_datas.extend(args_diff_datas)
                ret_datas.extend(ret_diff_datas)

    if not IS_CONSTRAINT:
        args_datas = random.sample(args_datas, int(len(args_datas) * 0.25))
        ret_datas = random.sample(ret_datas, int(len(ret_datas) * 0.2))

    return args_datas, ret_datas

def run():
    proj_usage_infos = ProjectUsageInfos()
    annotation_infos = {}
    recreate_data = False

    print(multiprocessing.cpu_count())
    cpu_count = min(multiprocessing.cpu_count(), 16)
    
    train_repo_path = Path.home() / "TypeT5" / "ManyTypes4Py" / "repos" / "train"
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
    if not recreate_data and os.path.isfile("train_data_all.parquet"):
        train_datas = pd.read_parquet('train_data_all.parquet', engine='pyarrow')
        ret_train_datas = pd.read_parquet('ret_train_data_all.parquet', engine='pyarrow')
    else:
        projects = list(train_repo_path.iterdir())

        print(f"Processing {len(projects)} projects...")

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

        # train_datas.to_parquet('train_data_all.parquet', index=False)
        ret_train_datas.to_parquet('ret_train_data_all.parquet', index=False)
        train_datas = pd.read_parquet('train_data_all.parquet', engine='pyarrow')
        ret_train_datas = pd.read_parquet('ret_train_data_all.parquet', engine='pyarrow')

    # ctx_ratio_col = train_datas['ctx_len'] / (train_datas['ctx_len'] + train_datas['target_ctx_len'])
    # train_datas['ctx_raio'] = ctx_ratio_col.round(2)
    # # Nan to -1
    # train_datas['ctx_raio'].fillna(-1, inplace=True)
    # target_ctx_ratio_col = train_datas['target_ctx_len'] / (train_datas['ctx_len'] + train_datas['target_ctx_len'])
    # train_datas['target_ctx_raio'] = target_ctx_ratio_col.round(2)
    # # Nan to -1
    # train_datas['target_ctx_raio'].fillna(-1, inplace=True)
    # train_datas.drop(columns=['total_len'], inplace=True)
    # train_cols = train_datas.columns.tolist()
    # cols = train_cols[:7] + ['ctx_raio', 'target_ctx_raio'] + train_cols[7:-2]
    # train_datas = train_datas[cols]


    # ret_train_cols = ret_train_datas.columns.tolist()
    # cols = ret_train_cols[3:]
    # ret_train_datas = ret_train_datas[cols]

    print(len(ret_train_datas))
    print(ret_train_datas.head(10))
    print(ret_train_datas.columns.tolist())


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

    test_repo_path = Path.home() / "TypeT5" / "ManyTypes4Py" / "repos" / "test"

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


        # test_datas.to_parquet('test_data_all.parquet', index=False)
        ret_test_datas.to_parquet('ret_test_data_all.parquet', index=False)
        test_datas = pd.read_parquet('test_data_all.parquet', engine='pyarrow')
        ret_test_datas = pd.read_parquet('ret_test_data_all.parquet', engine='pyarrow')

    # ctx_ratio_col = test_datas['ctx_len'] / (test_datas['ctx_len'] + test_datas['target_ctx_len'])
    # test_datas['ctx_raio'] = ctx_ratio_col.round(2)
    # # Nan to 0
    # test_datas['ctx_raio'].fillna(-1, inplace=True)
    # target_ctx_ratio_col = test_datas['target_ctx_len'] / (test_datas['ctx_len'] + test_datas['target_ctx_len'])
    # test_datas['target_ctx_raio'] = target_ctx_ratio_col.round(2)
    # # Nan to 0
    # test_datas['target_ctx_raio'].fillna(-1, inplace=True)
    # test_datas.drop(columns=['total_len'], inplace=True)
    # test_cols = test_datas.columns.tolist()
    # cols = test_cols[:7] + ['ctx_raio', 'target_ctx_raio'] + test_cols[7:-2]
    # test_datas = test_datas[cols]

    # ret_ctx_ratio_col = ret_test_datas['ctx_len'] / (ret_test_datas['ctx_len'] + ret_test_datas['target_ctx_len'])
    # ret_test_datas['ctx_raio'] = ret_ctx_ratio_col.round(2)
    # # Nan to 0
    # ret_test_datas['ctx_raio'].fillna(-1, inplace=True)
    # ret_target_ctx_ratio_col = ret_test_datas['target_ctx_len'] / (ret_test_datas['ctx_len'] + ret_test_datas['target_ctx_len'])
    # ret_test_datas['target_ctx_raio'] = ret_target_ctx_ratio_col.round(2)
    # # Nan to 0
    # ret_test_datas['target_ctx_raio'].fillna(-1, inplace=True)
    # ret_test_datas.drop(columns=['total_len'], inplace=True)
    # ret_test_cols = ret_test_datas.columns.tolist()
    # cols = ret_test_cols[3:7] + ['ctx_raio', 'target_ctx_raio'] + ret_test_cols[7:-2]
    # ret_test_datas = ret_test_datas[cols]



    print(len(ret_test_datas))
    print(ret_test_datas.head(10))
    print(ret_test_datas.columns.tolist())

    # exit()

    train(train_datas)
    get_test_accuracy(test_datas)

    train(ret_train_datas, is_return=True)
    get_test_accuracy(ret_test_datas, is_return=True)

if __name__ == "__main__":
    run()