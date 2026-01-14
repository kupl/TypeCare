import pickle
import os
from pathlib import Path
import json
import ast
from logger import logger
from collections import Counter, defaultdict
from argparse import ArgumentParser
import difflib
import glob
from pprint import pprint
import time
from utils import remove_prefix_filename, remove_suffix_filename, output_folder_name
from evaluation.print_table import PrintTable

from config import (
    DATA_PATH, 
    TYPET5_PREDICTION_PATH, 
    TYPET5_TRANSFORM_PATH, 
    TYPET5_OUTPUT_PATH,
    TIGER_RESULT_PATH, 
    TIGER_OUTPUT_PATH,
    TYPEGEN_TESTSET_PATH,
    TYPEGEN_OUTPUT_PATH,
    MODEL_PATH
)
    
from typet5.type_check import parse_type_str, PythonType, normalize_type


from evaluation.eval import Problem, ProblemList
from run.rerank import naive_rerank, rerank
from analysis.naive_static_analysis import analyze_code, ProjectUsageInfos



def get_function_arguments_from_string(function_str, code):
    tree = ast.parse(code)
    
    if '.' in function_str:
        class_path, function_name = function_str.rsplit('.', 1) 
        class_names = class_path.split('.')  
    else:
        class_names = []
        function_name = function_str
    

    def find_function_in_class(nodes, class_names, function_name):
        if len(class_names) == 0:  
            methods = []
            for node in nodes:
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    arguments = []
                    for arg in node.args.posonlyargs:
                        arguments.append(arg.arg)

                    for arg in node.args.args:
                        arguments.append(arg.arg)

                    if node.args.vararg:
                        arguments.append(f"*{node.args.vararg.arg}")
                    if node.args.kwarg:
                        arguments.append(f"**{node.args.kwarg.arg}")
                    methods.append(arguments)
                elif isinstance(node, (ast.If, ast.For, ast.While)):
                    methods.extend(find_function_in_class(node.body, class_names, function_name))
                elif isinstance(node, ast.ClassDef):
                    methods.extend(find_function_in_class(node.body, class_names, function_name))
            
            return methods
        else: 
            class_name = class_names[0]
            for node in nodes:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return find_function_in_class(node.body, class_names[1:], function_name)
        return []

    return find_function_in_class(tree.body, class_names, function_name)

def get_diff_by_remove(origin, remove):
    diff = []

    for result in origin:
        if result not in remove:
            diff.append(result)

    return diff

class StackNode:
    def __init__(self, name, is_function, parent=None):
        self.name = name
        self.is_function = is_function
        self.parent = parent

    def get_full_name(self):
        names = []
        node = self
        is_start = True
        while node:
            if not is_start and node.is_function:
                node = node.parent
            else:
                is_start = False
                names.append(node.name)
                node = node.parent

        return ".".join(reversed(names))

# Get arguments of the function (or method) signature from the given code
def get_arguments(code: str, target: str):
    tree = ast.parse(code)
    # 함수별 annotation 카운트를 저장할 defaultdict
    arguments_list = []

    def recursive_count_annotations(node, parent=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_node = StackNode(node.name, True, parent)
        elif isinstance(node, ast.ClassDef):
            current_node = StackNode(node.name, False, parent)
        else:
            current_node = parent
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and current_node.get_full_name() == target:
            arguments = []
            for arg in node.args.posonlyargs:
                arguments.append(arg.arg)

            for arg in node.args.args:
                arguments.append(arg.arg)

            if node.args.vararg:
                arguments.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                arguments.append(f"**{node.args.kwarg.arg}")

            nonlocal arguments_list
            arguments_list.append(arguments)


        for child in ast.iter_child_nodes(node):
            recursive_count_annotations(child, current_node)

    recursive_count_annotations(tree)
    return arguments_list

# check is property
def check_is_property(code: str, target: str):
    tree = ast.parse(code)
    is_property = False

    def recursive_check_property(node, parent=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_node = StackNode(node.name, True, parent)
        elif isinstance(node, ast.ClassDef):
            current_node = StackNode(node.name, False, parent)
        else:
            current_node = parent

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and current_node.get_full_name() == target:
            nonlocal is_property
            if node.decorator_list:
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "property":
                        is_property = True

        for child in ast.iter_child_nodes(node):
            recursive_check_property(child, current_node)

    recursive_check_property(tree)
    return is_property

def count_annotations(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # logger.error(f"Syntax error in code: {e}")
        return ({}, {})
    param_count = defaultdict(list)
    ret_count = defaultdict(list)

    def normalize_annotation(annotation):
        return str(parse_type_str(annotation))

    find_target = False

    def recursive_count_annotations(node, parent=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_node = StackNode(node.name, True, parent)
        elif isinstance(node, ast.ClassDef):
            current_node = StackNode(node.name, False, parent)
        else:
            current_node = parent

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_function = current_node.get_full_name()

            param_annotations = {}
            ret_annotation = None

            if node.returns:
                try:
                    ret_annotation = normalize_annotation(ast.unparse(node.returns))
                except:
                    pass


                # check if method is property
                is_property = False
                if node.decorator_list:
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name) and decorator.id == "property":
                            is_property = True

  
            for arg in node.args.args:
                if arg.annotation:
                    try:
                        annotation_type = normalize_annotation(ast.unparse(arg.annotation))
                        param_annotations[arg.arg] = annotation_type
                    except:
                        pass

            for arg in node.args.kwonlyargs:
                if arg.annotation:
                    try:
                        annotation_type = normalize_annotation(ast.unparse(arg.annotation))
                        param_annotations[arg.arg] = annotation_type
                    except:
                        pass


            param_counter = Counter()
            ret_counter = Counter()

            if ret_annotation:
                if is_property and False:
                    param_counter[node.name] = ret_annotation
                else:
                    ret_counter[node.name] = ret_annotation

            for arg, annotation in param_annotations.items():
                param_counter[arg] = annotation

            candidates = param_count[current_function]
            candidates.append(param_counter)
            param_count[current_function] = candidates

            candidates = ret_count[current_function]
            candidates.append(ret_counter)
            ret_count[current_function] = candidates

        for child in ast.iter_child_nodes(node):
            recursive_count_annotations(child, current_node)

    recursive_count_annotations(tree)
    return param_count, ret_count
    

    return arg_counts, ret_counts


def get_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def group_similar_variables(variables, threshold=0.8):
    grouped = []
    visited = set()
    
    for i in range(len(variables)):
        if variables[i] in visited:
            continue
        group = set([variables[i]])
        visited.add(variables[i])
        
        for j in range(i + 1, len(variables)):
            if get_similarity(variables[i], variables[j]) >= threshold:
                group.add(variables[j])
                visited.add(variables[j])
        
        grouped.append(group)
    
    return grouped

# 주석에 대한 annotation 수를 세는 함수
def get_annotation_count(group, annotations):
    annots = Counter()
    for var in group:
        for annotation in annotations:
            if var in annotation:
                annots[annotation[var]] += 1
    return annots

class TypeT5Analysis:
    problem_list: ProblemList

    def __init__(self):
        self.problem_list = self.load_transformed_data()

    def export_fail_json(self):
        fail_json = {}
        for problem in self.problem_list.get_problem_list():
            result_path = problem.proj_result_path

            if problem.after_solutions[0] in problem.correct_num_list:
                continue

            first_num = problem.after_solutions[0]

            try:
                params = problem.params
                preds = problem.preds[first_num]
                expects = problem.expects
            except Exception as e:
                raise e

            data = {}
            for param, pred, expect in zip(params, preds, expects):
                data[str(param)] = {
                    "pred": str(pred),
                    "expect": str(expect)
                }

            fail_json[str(result_path)] = data

        with open("fail.json", 'w') as f:
            json.dump(fail_json, f, indent=4)

    def get_result_path(self, repo_name, file_path, target):
        folder_name = output_folder_name(repo_name, file_path)
        path = DATA_PATH / "TypeT5_1118" / folder_name / str(target).replace(".", "_")
        if not path.exists():
            folder_name = output_folder_name(repo_name, "src/" + file_path)
            path = DATA_PATH / "TypeT5_1118" / folder_name / target

        if not path.exists():
            return None
    
        return path

    def load_transformed_data(self):
        error_num = 0
        logger.info("Loading TypeT5 transformed prediction results...")
        problem_list = ProblemList()

        with open(TYPET5_TRANSFORM_PATH, "r") as f:
            typet5_transform = json.load(f)
        
        logger.info("TypeT5 transformed prediction results loaded: %i projects", len(typet5_transform))

        for i, data in enumerate(typet5_transform):
            if i % 1000 == 0:
                logger.info(f"({i+1}/{len(typet5_transform)}) Processing {data['repo_name']}")



            repo_name = data['repo_name']
            # src_path = data['src_path']
            file_path = data['file_path']

            # proj_result_path = Path(data['result_path'])
            target = data['target']
            params = data['params']
            preds = data['predictions']
            expects = data['expects']
            result_path = self.get_result_path(repo_name, file_path, target)

            if result_path is None:
                continue
            
            file_info_json = result_path / "file_info.json"
            info_json = result_path / "info.json"

            if not os.path.exists(info_json):
                error_num += 1
                continue


            with open(info_json, 'r') as f:
                info = json.load(f)
            
            correct_num_list = info['correct']
            assert len(preds[0]) == len(expects)

            expect_len = len(expects)
            modified_preds = [ pred[:expect_len] for pred in preds]

            # drop param and preds if element of expects is None
            none_idx = [i for i, expect in enumerate(expects) if expect is None]
            expects = [expect for i, expect in enumerate(expects) if i not in none_idx]
            preds = []
            for pred in modified_preds:
                preds.append([x for i, x in enumerate(pred) if i not in none_idx])
            # preds = [pred for i, pred in enumerate(modified_preds) if i not in none_idx]
            params = [param for i, param in enumerate(params) if i not in none_idx]


            problem = Problem(repo_name, file_path, result_path, target, correct_num_list, params, preds, expects)
            
            before_solutions = [0,1,2,3,4,5,6,7,8,9]
            problem.set_before_solutions(before_solutions)

            problem_list.add_problem(problem)
        return problem_list

    def get_after_solutions(self):
        annotations_info = {}
        proj_usage_infos = ProjectUsageInfos()

        total_time = 0
        infer_time = 0

        time_dict = {}

        with open(MODEL_PATH / 'random_forest_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open(MODEL_PATH / 'ret_random_forest_model.pkl', 'rb') as f:
            ret_clf = pickle.load(f)

        
        print("Model loaded from file.")

        for i, problem in enumerate(self.problem_list.get_problem_list()):
            if i % 500 == 0:
                logger.info(f"({i+1}/{len(self.problem_list.get_problem_list())}) Processing...")

            start_time = time.time()

            proj_path = Path("BetterTypes4Py/repos/test") / problem.repo_name
            file_path = problem.file_path

            if not os.path.exists(proj_path / file_path):
                file_path = "src/" + file_path

            func_name = problem.target
            result_path = self.get_result_path(problem.repo_name, problem.file_path, problem.target)

            removed_json = result_path / "removed.json"

            if not os.path.exists(removed_json):
                # logger.error("Correct json file not found: %s", correct_json)
                continue

            file_start_time = time.time()

            # if "_record_session" not in str(result_path):
            #     continue


            with open(removed_json, 'r') as f:
                removed_json = json.load(f)['generalDiagnostics']

            for rem in removed_json:
                filename = remove_prefix_filename(rem['file'])
                filename = remove_suffix_filename(filename, is_removed=True)
                rem['file'] = filename

            analysis_results = []

            # correct_diff = get_diff_by_remove(correct_json, removed_json)

            for i in range(10):
                # if i in problem.correct_num_list:
                #     analysis_results.append(correct_diff)
                # else:
                modified_json_file = result_path / f"modified_{i}.json"
                try:
                    with open(modified_json_file, 'r') as f:
                        modified_json = json.load(f)['generalDiagnostics']
                    for mod in modified_json:
                        filename = remove_prefix_filename(mod['file'])
                        filename = remove_suffix_filename(filename, is_removed=False)
                        mod['file'] = filename
                except Exception as e:
                    # logger.error(f"Error loading modified json file: {modified_json}")
                    # logger.error(e)
                    continue

                modified_diff = get_diff_by_remove(modified_json, removed_json)

                analysis_results.append(modified_diff)

            

            if file_path in annotations_info:
                annotations = annotations_info[file_path]
            else:
                with open(proj_path / file_path, 'r') as f:
                    code = f.read()
                annotations = count_annotations(code)
                annotations_info[file_path] = annotations

            # print(file_path)
            # print(proj_path)
            # input()
            # continue

            proj_usage_infos.update_usage_infos(proj_path)


            # Annotation Info
            param_count, ret_count = annotations

            param_counter_list = [
                counter_list for func, counter_list in param_count.items() 
                if func != problem.target # and func.split('.')[:-1] == problem.target.split('.')[:-1]
            ]

            ret_counter_list = [
                counter_list for func, counter_list in ret_count.items() 
                if func != problem.target # and func.split('.')[:-1] == problem.target.split('.')[:-1]
            ]

            param_counter = defaultdict(Counter)
            ret_counter = defaultdict(Counter)

            for annot_list in param_counter_list:
                for annot in annot_list:
                    for var, typ in annot.items():
                        param_counter[var][typ] += 1

            for annot_list in ret_counter_list:
                for annot in annot_list:
                    for var, typ in annot.items():
                        ret_counter[var][typ] += 1

            params = problem.params

            param_ctx_counter = defaultdict(dict)
            ret_ctx_counter = defaultdict(dict)

            target_split = func_name.split('.')
            target_class = '.'.join(target_split[:-1])
            target_func = target_split[-1]

            ctx_types = proj_usage_infos.extract_ctx_types(proj_path, file_path, target_class, target_func)
            target_ctx_types = proj_usage_infos.extract_target_ctx_types(proj_path, file_path, target_class, target_func)



            is_property = False

            start_infer_time = time.time()

            after_solutions, correct_num = rerank(proj_path, file_path, result_path, func_name, problem.expects, problem.before_solutions, analysis_results, problem.preds, params, param_counter, ret_counter, is_property, ctx_types, target_ctx_types, problem.total_preds, clf, ret_clf)
            
            end_infer_time = time.time()
            infer_time += end_infer_time - start_infer_time
            
            after_solutions_num = [sol.num for sol in after_solutions]

            problem.set_after_solutions(after_solutions_num)

            before_correct_num_list = problem.correct_num_list
            for i in correct_num:
                problem.add_correct_num(i)

            for sol in after_solutions:
                if sol.num >= 10 or sol.num < 0:
                    problem.add_pred(sol.num, sol.pred)

            end_time = time.time()
            total_time += end_time - start_time
            file_time = end_time - file_start_time

            time_dict[str(result_path)] = file_time

            if not after_solutions:
                continue
            if not analysis_results:
                continue

        # with open("typet5_time.txt", 'w') as f:
        #     f.write(f"Total Time: {total_time}\n")
        #     f.write(f"Infer Time: {infer_time}\n")
        #     f.write(f"Average Time: {total_time / len(self.problem_list.get_problem_list())}\n")
        #     f.write(f"Average Infer Time: {infer_time / len(self.problem_list.get_problem_list())}\n")

        with open("typet5_time_dict.json", 'w') as f:
            json.dump(time_dict, f, indent=4)



    def calc_before_top_k(self):
        return self.problem_list.calc_before_top_k()
        
    def calc_base_before_top_k(self):
        return self.problem_list.calc_base_before_top_k()

    def calc_after_top_k(self):
        return self.problem_list.calc_after_top_k()

    def calc_base_after_top_k(self):
        return self.problem_list.calc_base_after_top_k()

    def return_top_k(self):
        return self.problem_list.return_top_k()

    def print_top_k(self):
        self.problem_list.print_top_k()

    def return_base_top_k(self):
        return self.problem_list.return_base_top_k()

    def print_base_top_k(self):
        self.problem_list.print_base_top_k()

    def return_arg_ret_top_k(self):
        return self.problem_list.return_arg_ret_top_k()

    def print_arg_ret_top_k(self):
        self.problem_list.print_arg_ret_top_k()


class TypeGenAnalysis:
    problem_list: ProblemList

    def __init__(self):
        self.problem_list = self.load_transformed_data()

    def load_transformed_data(self):
        error_num = 0
        logger.info("Loading TypeGen transformed prediction results...")
        problem_list = ProblemList()

        # typegen_path = Path.home() / "TypeGen"
        result_path = TYPEGEN_TESTSET_PATH


        with open(result_path, "r") as f:
            typegen_results = json.load(f)

        logger.info("TypeGen transformed prediction results loaded: %i projects", len(typegen_results))

        for i, data in enumerate(typegen_results):
            if i % 1000 == 0:
                logger.info(f"({i+1}/{len(typegen_results)}) Processing {data['repo_name']}")

            repo_name = data['repo_name']
            src_path = data['src_path']

            proj_result_path = Path(data['result_path'])
            target = data['target']
            params = data['params']
            preds = data['predictions']
            expects = data['expects']

            cat = data['cat']
            generic = data['generic']
            total_preds = data['total_predictions']

            

            result_path = DATA_PATH / "TypeGen" / proj_result_path
            
            file_info_json = result_path / "file_info.json"
            info_json = result_path / "info.json"

            if not os.path.exists(info_json):
                error_num += 1
                continue


            with open(info_json, 'r') as f:
                info = json.load(f)

            with open(file_info_json, 'r') as f:
                file_info = json.load(f)

            
            file_path = file_info['file_path']
            correct_num_list = info['correct']

            expect_len = len(expects)
            modified_preds = preds

            # drop param and preds if element of expects is None
            none_idx = [i for i, expect in enumerate(expects) if expect is None]
            expects = [str(parse_type_str(expect)) for i, expect in enumerate(expects) if i not in none_idx]
            preds = []
            for pred in modified_preds:
                new_pred = []
                for i, x in enumerate(pred):
                    if i in none_idx:
                        continue

                    try:
                        modified_type = str(parse_type_str(x))
                    except Exception as e:
                        modified_type = x

                    new_pred.append(modified_type)

                preds.append(new_pred)

            # preds = [pred for i, pred in enumerate(modified_preds) if i not in none_idx]
            params = [param for i, param in enumerate(params) if i not in none_idx]

            problem = Problem(repo_name, src_path, file_path, proj_result_path, target, correct_num_list, params, preds, expects, cat=cat, generic=generic, total_preds=total_preds)
            
            prob_len = min(len(problem.preds), 10)

            before_solutions = [i for i in range(prob_len)]
            problem.set_before_solutions(before_solutions)

            problem_list.add_problem(problem)
        return problem_list

    def get_after_solutions(self):
        annotations_info = {}
        proj_usage_infos = ProjectUsageInfos()

        total_time = 0
        infer_time = 0

        time_dict = {}

        with open(MODEL_PATH / 'many_random_forest_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open(MODEL_PATH / 'many_ret_random_forest_model.pkl', 'rb') as f:
            ret_clf = pickle.load(f)

        for i, problem in enumerate(self.problem_list.get_problem_list()):
            if i % 500 == 0:
                logger.info(f"({i+1}/{len(self.problem_list.get_problem_list())}) Processing...")

            start_time = time.time()

            proj_result_path = problem.proj_result_path
            src_path = problem.src_path

            proj_idx = problem.file_path.find(src_path)
            proj_path = Path.home() / "TypeGen" / problem.src_path
            file_path = Path.home() / "TypeGen" / problem.file_path


            # if file_path != "/home/wonseokoh/TypeT5/ManyTypes4Py/repos/test/marcosschroh__dataclasses-avroschema/dataclasses_avroschema/pydantic/fields.py":
            #     continue
            # if "avro_type" not in problem.target:
            #     continue
            func_name = problem.target
            result_path = DATA_PATH / "TypeGen" / proj_result_path

            removed_json = result_path / "removed.json"

            if not os.path.exists(removed_json):
                # logger.error("Correct json file not found: %s", removed_json)
                continue

            file_start_time = time.time()

            with open(removed_json, 'r') as f:
                removed_json = json.load(f)['generalDiagnostics']

            for rem in removed_json:
                filename = remove_prefix_filename(rem['file'])
                filename = remove_suffix_filename(filename, is_removed=True)
                rem['file'] = filename

            analysis_results = []


            for i in range(len(problem.preds)):
                try:
                    modified_json = result_path / f"modified_{i}.json"
                    with open(modified_json, 'r') as f:
                        modified_json = json.load(f)['generalDiagnostics']

                    for mod in modified_json:
                        filename = remove_prefix_filename(mod['file'])
                        filename = remove_suffix_filename(filename, is_removed=False)
                        mod['file'] = filename


                    modified_diff = get_diff_by_remove(modified_json, removed_json)

                    

                    analysis_results.append(modified_diff)
                except FileNotFoundError:
                    # logger.error("Modified json file not found: %s", modified_json)
                    # analysis_results.append(None)
                    continue

            if file_path in annotations_info:
                annotations = annotations_info[file_path]
            else:
                with open(file_path, 'r') as f:
                    code = f.read()
                annotations = count_annotations(code)
                annotations_info[file_path] = annotations

            param_count, ret_count = annotations

            param_counter_list = [
                counter_list for func, counter_list in param_count.items() 
                if func != problem.target # and func.split('.')[:-1] == problem.target.split('.')[:-1]
            ]

            ret_counter_list = [
                counter_list for func, counter_list in ret_count.items()
                if func != problem.target # and func.split('.')[:-1] == problem.target.split('.')[:-1]
            ]


            param_counter = defaultdict(Counter)
            ret_counter = defaultdict(Counter)

            for annot_list in param_counter_list:
                for annot in annot_list:
                    for var, typ in annot.items():
                        param_counter[var][typ] += 1

            for annot_list in ret_counter_list:
                for annot in annot_list:
                    for var, typ in annot.items():
                        ret_counter[var][typ] += 1

            params = problem.params

            param_ctx_counter = defaultdict(dict)
            ret_ctx_counter = defaultdict(dict)

            target_split = func_name.split('.')
            target_class = '.'.join(target_split[:-1])
            target_func = target_split[-1]

            ctx_types = proj_usage_infos.extract_ctx_types(proj_path, file_path, target_class, target_func)
            target_ctx_types = proj_usage_infos.extract_target_ctx_types(proj_path, file_path, target_class, target_func)

            is_property = False

            err_idx = []
            for idx, analysis_result in enumerate(analysis_results):
                if analysis_result is None:
                    err_idx.append(idx)

            modified_before_solutions = [sol for i, sol in enumerate(problem.before_solutions) if i not in err_idx]
            modified_analysis_results = [sol for i, sol in enumerate(analysis_results) if i not in err_idx]
            modified_preds = [sol for i, sol in enumerate(problem.preds) if i not in err_idx]

            problem.set_before_solutions(modified_before_solutions)

            if modified_before_solutions == []:
                end_time = time.time()
                total_time += end_time - start_time
                time_dict[str(result_path)] = end_time - file_start_time
                continue

            start_infer_time = time.time()

            after_solutions, correct_num = rerank(proj_path, file_path, result_path, func_name, problem.expects, problem.before_solutions, analysis_results, problem.preds, params, param_counter, ret_counter, is_property, ctx_types, target_ctx_types, problem.total_preds, clf, ret_clf)
            
            end_infer_time = time.time()
            infer_time += end_infer_time - start_infer_time
            
            after_solutions_num = [sol.num for sol in after_solutions]
            
            problem.set_after_solutions(after_solutions_num)

            before_correct_num_list = problem.correct_num_list
            for i in correct_num:
                problem.add_correct_num(i)

            for sol in after_solutions:
                if sol.num >= 10:
                    problem.add_pred(sol.num, sol.pred)

            end_time = time.time()
            total_time += end_time - start_time
            file_time = end_time - file_start_time

            time_dict[str(result_path)] = file_time

            if not after_solutions:
                continue
            if not analysis_results:
                continue

        # with open("typegen_time.txt", 'w') as f:
        #     f.write(f"Total Time: {total_time}\n")
        #     f.write(f"Infer Time: {infer_time}\n")
        #     f.write(f"Average Time: {total_time / len(self.problem_list.get_problem_list())}\n")
        #     f.write(f"Average Infer Time: {infer_time / len(self.problem_list.get_problem_list())}\n")

        with open("typegen_time_dict.json", 'w') as f:
            json.dump(time_dict, f, indent=4)

    def calc_before_top_k(self):
        return self.problem_list.calc_before_top_k()

    def calc_after_top_k(self):
        return self.problem_list.calc_after_top_k()

    def return_top_k(self):
        return self.problem_list.return_top_k()

    def print_top_k(self):
        self.problem_list.print_top_k()

    def return_base_top_k(self):
        return self.problem_list.return_base_top_k()

    def print_base_top_k(self):
        self.problem_list.print_base_top_k()

    def return_categorized_top_k(self):
        return self.problem_list.return_categorized_top_k()

    def print_categorized_top_k(self):
        self.problem_list.print_categorized_top_k()

    def return_arg_ret_top_k(self):
        return self.problem_list.return_arg_ret_top_k()

    def print_arg_ret_top_k(self):
        self.problem_list.print_arg_ret_top_k()

    def var_to_type_dict(self):
        var_to_type = self.problem_list.var_to_type_dict()

        # sort by value

        var_to_type = {k: v for k, v in sorted(var_to_type.items(), key=lambda item: len(item[1]["type_set"]))}
        with open("var_to_type.json", 'w') as f:
            json.dump(var_to_type, f, indent=4)


    def export_fail_json(self):
        fail_json = {}
        for problem in self.problem_list.get_problem_list():
            result_path = problem.proj_result_path

            if not problem.after_solutions:
                continue

            if problem.after_solutions[0] in problem.correct_num_list:
                continue

            first_num = problem.after_solutions[0]

            try:
                params = problem.params
                preds = problem.preds[first_num]
                expects = problem.expects
            except Exception as e:
                print(first_num)
                print(problem.preds)
                print(problem.correct_num_list)
                raise e

            data = {}
            for param, pred, expect in zip(params, preds, expects):
                data[str(param)] = {
                    "pred": str(pred),
                    "expect": str(expect)
                }

            fail_json[str(result_path)] = data

        with open("typegen_fail.json", 'w') as f:
            json.dump(fail_json, f, indent=4)


class TigerAnalysis:
    tool_name = "Tiger"
    problem_list: ProblemList

    def __init__(self):
        self.problem_list = self.load_transformed_data()

    def change_test_info(self, test_info):
        src_path = test_info["src_path"]
        file_path = test_info["file_path"]

        file_path = file_path[file_path.find(src_path)+len(src_path)+1:]

        test_info["file_path"] = file_path
        return test_info

    def get_result_path(self, repo_name, file_path, target, param):
        folder_name = output_folder_name(repo_name, file_path)
        path = DATA_PATH / "Tiger" / folder_name / target / param
        if not path.exists():
            folder_name = output_folder_name(repo_name, "src/" + file_path)
            path = DATA_PATH / "Tiger" / folder_name / target / param
        
        if not path.exists():
            return None
    
        return path

    def load_transformed_data(self):
        error_num = 0
        logger.info("Loading Tiger transformed prediction results...")
        problem_list = ProblemList()

        # tiger_path = Path.home() / "TypeInfer-Replication"
        result_path = TIGER_RESULT_PATH



        with open(result_path, "r") as f:
            tiger_results = json.load(f)

        logger.info("Tiger transformed prediction results loaded: %i projects", len(tiger_results))

        for i, data in enumerate(tiger_results):
            if i % 1000 == 0:
                logger.info(f"({i+1}/{len(tiger_results)}) Processing {data['repo_name']}")

            data = self.change_test_info(data)

            repo_name = data['repo_name']
            src_path = data['src_path']
            file_path = data["file_path"]

            proj_result_path = Path(data['result_path'])
            target = data['target']
            params = data['params']
            preds = data['predictions']
            expects = data['expects']

            cat = data['cat']
            generic = data['generic']
            total_preds = data['total_predictions']

            result_path = self.get_result_path(repo_name, file_path, target, params[0])
            if result_path is None:
                error_num += 1
                continue

            
            file_info_json = result_path / "file_info.json"
            info_json = result_path / "info.json"

            if not os.path.exists(info_json):
                error_num += 1
                continue


            with open(info_json, 'r') as f:
                info = json.load(f)

            correct_num_list = info['correct']
            modified_preds = preds



            # drop param and preds if element of expects is None
            none_idx = [i for i, expect in enumerate(expects) if expect is None]
            expects = [str(parse_type_str(expect)) for i, expect in enumerate(expects) if i not in none_idx]
            preds = []
            for pred in modified_preds:
                new_pred = []
                for i, x in enumerate(pred):
                    if i in none_idx:
                        continue

                    try:
                        modified_type = str(parse_type_str(x))
                    except Exception as e:
                        modified_type = x

                    new_pred.append(modified_type)

                preds.append(new_pred)

            # preds = [pred for i, pred in enumerate(modified_preds) if i not in none_idx]
            params = [param for i, param in enumerate(params) if i not in none_idx]

            problem = Problem(repo_name, src_path, file_path, proj_result_path, target, correct_num_list, params, preds, expects, cat=cat, generic=generic, total_preds=total_preds)
            
            prob_len = min(len(problem.preds), 10)

            before_solutions = [i for i in range(prob_len)]
            problem.set_before_solutions(before_solutions)

            problem_list.add_problem(problem)
        return problem_list

    def get_after_solutions(self):
        annotations_info = {}
        proj_usage_infos = ProjectUsageInfos()

        total_time = 0
        infer_time = 0

        time_dict = {}

        with open(MODEL_PATH / 'many_random_forest_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open(MODEL_PATH / 'many_ret_random_forest_model.pkl', 'rb') as f:
            ret_clf = pickle.load(f)

        for i, problem in enumerate(self.problem_list.get_problem_list()):
            if i % 500 == 0:
                logger.info(f"({i+1}/{len(self.problem_list.get_problem_list())}) Processing...")

            proj_result_path = problem.proj_result_path
            src_path = problem.src_path

            proj_idx = problem.file_path.find(src_path)
            proj_path = Path("ManyTypes4Py") / problem.src_path
            file_path = problem.file_path


            func_name = problem.target
            result_path = self.get_result_path(problem.repo_name, problem.file_path, problem.target, problem.params[0])
            if result_path is None:
                continue

            removed_json = result_path / "removed.json"

            if not os.path.exists(removed_json):
                # logger.error("Removed json file not found: %s", removed_json)
                continue

            file_start_time = time.time()

            with open(removed_json, 'r') as f:
                removed_json = json.load(f)['generalDiagnostics']

            for rem in removed_json:
                filename = remove_prefix_filename(rem['file'])
                filename = remove_suffix_filename(filename, is_removed=True)
                rem['file'] = filename

            analysis_results = []

            start_time = time.time()

            for i in range(len(problem.preds)):
                try:
                    modified_json = result_path / f"modified_{i}.json"
                    with open(modified_json, 'r') as f:
                        modified_json = json.load(f)['generalDiagnostics']

                    for mod in modified_json:
                        filename = remove_prefix_filename(mod['file'])
                        filename = remove_suffix_filename(filename, is_removed=False)
                        mod['file'] = filename


                    modified_diff = get_diff_by_remove(modified_json, removed_json)

                    analysis_results.append(modified_diff)
                except FileNotFoundError:
                    # logger.error("Modified json file not found: %s", modified_json)
                    # analysis_results.append(None)
                    continue

            if file_path in annotations_info:
                annotations = annotations_info[file_path]
            else:
                with open(proj_path / file_path, 'r') as f:
                    code = f.read()
                annotations = count_annotations(code)
                annotations_info[file_path] = annotations

            param_count, ret_count = annotations

            param_counter_list = [
                counter_list for func, counter_list in param_count.items() 
                if func != problem.target # and func.split('.')[:-1] == problem.target.split('.')[:-1]
            ]

            ret_counter_list = [
                counter_list for func, counter_list in ret_count.items()
                if func != problem.target # and func.split('.')[:-1] == problem.target.split('.')[:-1]
            ]


            param_counter = defaultdict(Counter)
            ret_counter = defaultdict(Counter)

            for annot_list in param_counter_list:
                for annot in annot_list:
                    for var, typ in annot.items():
                        param_counter[var][typ] += 1

            for annot_list in ret_counter_list:
                for annot in annot_list:
                    for var, typ in annot.items():
                        ret_counter[var][typ] += 1

            params = problem.params

            target_split = func_name.split('.')
            target_class = '.'.join(target_split[:-1])
            target_func = target_split[-1]

            ctx_types = proj_usage_infos.extract_ctx_types(proj_path, file_path, target_class, target_func)
            target_ctx_types = proj_usage_infos.extract_target_ctx_types(proj_path, file_path, target_class, target_func)

            is_property = False

            err_idx = []
            for idx, analysis_result in enumerate(analysis_results):
                if analysis_result is None:
                    err_idx.append(idx)

            modified_before_solutions = [sol for i, sol in enumerate(problem.before_solutions) if i not in err_idx]

            problem.set_before_solutions(modified_before_solutions)

            if modified_before_solutions == []:
                end_time = time.time()
                total_time += end_time - start_time
                time_dict[str(result_path)] = end_time - file_start_time
                continue

            start_infer_time = time.time()

            after_solutions, correct_num = rerank(proj_path, file_path, result_path, func_name, problem.expects, problem.before_solutions, analysis_results, problem.preds, params, param_counter, ret_counter, is_property, ctx_types, target_ctx_types, problem.total_preds, clf, ret_clf)
            
            end_infer_time = time.time()
            infer_time += end_infer_time - start_infer_time

            after_solutions_num = [sol.num for sol in after_solutions]
            
            problem.set_after_solutions(after_solutions_num)

            before_correct_num_list = problem.correct_num_list
            for i in correct_num:
                problem.add_correct_num(i)

            for sol in after_solutions:
                if sol.num >= 10:
                    problem.add_pred(sol.num, sol.pred)

            end_time = time.time()
            total_time += end_time - start_time
            file_time = end_time - file_start_time

            time_dict[str(result_path)] = file_time

            if not after_solutions:
                continue
            if not analysis_results:
                continue

        # with open("tiger_time.txt", 'w') as f:
        #     f.write(f"Total time: {total_time:.2f} seconds\n")
        #     f.write(f"Infer time: {infer_time:.2f} seconds\n")
        #     f.write(f"Average time per problem: {total_time / len(self.problem_list.get_problem_list()):.2f} seconds\n")
        #     f.write(f"Average infer time per problem: {infer_time / len(self.problem_list.get_problem_list()):.2f} seconds\n")

        with open("tiger_time_dict.json", 'w') as f:
            json.dump(time_dict, f, indent=4)

    def calc_before_top_k(self):
        return self.problem_list.calc_before_top_k()

    def calc_after_top_k(self):
        return self.problem_list.calc_after_top_k()

    def return_top_k(self):
        return self.problem_list.return_top_k()

    def print_top_k(self):
        self.problem_list.print_top_k()

    def return_base_top_k(self):
        return self.problem_list.return_base_top_k()

    def print_base_top_k(self):
        self.problem_list.print_base_top_k()

    def return_categorized_top_k(self):
        return self.problem_list.return_categorized_top_k()

    def print_categorized_top_k(self):
        self.problem_list.print_categorized_top_k()

    def return_arg_ret_top_k(self):
        return self.problem_list.return_arg_ret_top_k()

    def print_arg_ret_top_k(self):
        self.problem_list.print_arg_ret_top_k()

    def var_to_type_dict(self):
        var_to_type = self.problem_list.var_to_type_dict()

        # sort by value

        var_to_type = {k: v for k, v in sorted(var_to_type.items(), key=lambda item: len(item[1]["type_set"]))}
        with open("tiger_var_to_type.json", 'w') as f:
            json.dump(var_to_type, f, indent=4)


    def export_fail_json(self):
        fail_json = {}
        for problem in self.problem_list.get_problem_list():
            result_path = problem.proj_result_path

            if not problem.after_solutions:
                continue

            if problem.after_solutions[0] in problem.correct_num_list:
                continue

            first_num = problem.after_solutions[0]

            # print(first_num)
            # print(problem.preds)
            # print(len(problem.preds))

            try:
                params = problem.params
                preds = problem.preds[first_num]
                expects = problem.expects
            except Exception as e:
                print(first_num)
                print(problem.preds)
                print(problem.correct_num_list)
                raise e

            data = {}
            for param, pred, expect in zip(params, preds, expects):
                data[str(param)] = {
                    "pred": str(pred),
                    "expect": str(expect)
                }

            fail_json[str(result_path)] = data

        with open("tiger_fail.json", 'w') as f:
            json.dump(fail_json, f, indent=4)


def debug_typet5():
    typet5_analysis = TypeT5Analysis()
    typet5_analysis.debug()

def analysis_typet5(is_evaluate):
    typet5_analysis = TypeT5Analysis()
    typet5_analysis.get_after_solutions()

    with open(TYPET5_OUTPUT_PATH, 'wb') as f:
        pickle.dump(typet5_analysis, f)

    print("=== Function Signature Top k ===")
    typet5_analysis.print_top_k()
    print()
    print("=== Base Function Signature Top k ===")
    typet5_analysis.print_base_top_k()
    typet5_analysis.print_arg_ret_top_k()


def analysis_typet5(is_evaluate):
    typet5_analysis = TypeT5Analysis()
    typet5_analysis.get_after_solutions()

    with open(TYPET5_OUTPUT_PATH, 'wb') as f:
        pickle.dump(typet5_analysis, f)

    print("=== Function Signature Top k ===")
    typet5_analysis.print_top_k()
    print()
    print("=== Base Function Signature Top k ===")
    typet5_analysis.print_base_top_k()
    typet5_analysis.print_arg_ret_top_k()

def evaluate_typet5():
    with open(TYPET5_OUTPUT_PATH, 'rb') as f:
        typet5_analysis = pickle.load(f)

    print("=== Function Signature Top k ===")
    typet5_analysis.print_top_k()
    print()
    print("=== Base Function Signature Top k ===")
    typet5_analysis.print_base_top_k()
    typet5_analysis.print_arg_ret_top_k()


def analysis_typegen(is_evaluate):
    typegen_analysis = TypeGenAnalysis()
    typegen_analysis.get_after_solutions()

    with open(TYPEGEN_OUTPUT_PATH, 'wb') as f:
        pickle.dump(typegen_analysis, f)


    print("Top K")
    typegen_analysis.print_top_k()
    print()
    print("Base Top K")
    typegen_analysis.print_base_top_k()
    print()
    typegen_analysis.print_categorized_top_k()
    print()
    typegen_analysis.print_arg_ret_top_k()


def analysis_tiger():
    tiger_analysis = TigerAnalysis()
    tiger_analysis.get_after_solutions()

    with open(TIGER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(tiger_analysis, f)

    print("Top K")
    tiger_analysis.print_top_k()
    print()
    print("Base Top K")
    tiger_analysis.print_base_top_k()
    print()
    tiger_analysis.print_categorized_top_k()
    print()
    tiger_analysis.print_arg_ret_top_k()


def evaluate():
    with open(TYPET5_OUTPUT_PATH, 'rb') as f:
        typet5_analysis = pickle.load(f)

    with open(TYPEGEN_OUTPUT_PATH, 'rb') as f:
        typegen_analysis = pickle.load(f)

    with open(TIGER_OUTPUT_PATH, 'rb') as f:
        tiger_analysis = pickle.load(f)

    table = PrintTable(typet5_analysis, typegen_analysis, tiger_analysis)

    print("=== Main Table ===")
    table.print_main_table()
    print()
    print("=== Function Signature Table ===")
    table.print_signature_table()
    print()
    print("=== Param and Return Type Table ===")
    table.print_param_return_table()
    print()
    print("=== Categorized Table ===")
    table.print_categorized_table()

    



if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--tool", "-t", type=str, default="typet5")
    argument_parser.add_argument("--all", "-a", action="store_true", help="Run all analysis tools")
    argument_parser.add_argument("--evaluate", "-e", action="store_true", help="Evaluate the analysis results")

    args = argument_parser.parse_args()

    if args.evaluate:
        print(args.tool)
        if args.tool == "typet5":
            evaluate_typet5()
        else:
            evaluate()
        exit(0)

    if args.all:
        analysis_typet5(args.evaluate)
        analysis_typegen(args.evaluate)
        analysis_tiger(args.evaluate)
    else:
        if args.tool == "typet5":
            analysis_typet5(args.evaluate)
        elif args.tool == "typegen":
            analysis_typegen(args.evaluate)
        elif args.tool == "tiger":
            analysis_tiger(args.evaluate)
