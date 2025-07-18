from dataclasses import dataclass
from functools import cmp_to_key
from logger import logger
import json
from collections import Counter
from difflib import get_close_matches
from typet5.type_check import parse_type_str, normalize_type
import re
from copy import deepcopy
import math
import pickle
import os
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher
from .make_data import make_data
import numpy as np
import ast
from typing import List, Optional, Tuple
from shared_data import get_ast_cache
from copy import copy
from multiprocessing import Pool
import shutil
import time
from utils import remove_prefix_filename, remove_suffix_filename

from .run_static_analysis import run_static_analysis_typet5

STATIC_CLASSIFICATION = False

ONLY_STATIC_ANALYSIS = False
STATIC_ANALYSIS = True
CONTEXT_SCORE = True
CONTEXT_CANDIDATE = True
USE_CACHE = True

def collect_returns(tree: ast.AST) -> List[ast.Return]:
    return_nodes = []

    class ReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return):
            # if isinstance(node.value, ast.Name):
            return_nodes.append(node)
            self.generic_visit(node)

    ReturnVisitor().visit(tree)
    return return_nodes

def collect_assigns_for_return(tree: ast.AST, target_name: str, target_return: ast.Return) -> List[ast.Assign]:
    assigns = []
    is_find = False

    def walk_block(block: List[ast.stmt], available_assigns: List[ast.Assign]) -> None:
        nonlocal is_find, assigns
                
        if is_find:
            return []
        
        local_assigns = available_assigns.copy()

        for stmt in block:
            if is_find:
                return []

            if stmt is target_return:
                # Return reached — resolve assignments up to this point
                assigns = local_assigns
                is_find = True
                return []
            
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    # Check if the assignment is a simple variable assignment
                    var_name = stmt.targets[0].id
                    if var_name == target_name:
                        # If the variable name matches, add it to the list
                        local_assigns = [stmt]
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    var_name = stmt.target.id
                    if var_name == target_name:
                        # If the variable name matches, add it to the list
                        local_assigns = [stmt]

            elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With)):
                has_orelse = False
                candidate_assigns = []
                if hasattr(stmt, 'body'): 
                    candidate_assigns += walk_block(stmt.body, local_assigns)
                if hasattr(stmt, 'orelse'):
                    has_orelse = True
                    candidate_assigns += walk_block(stmt.orelse, local_assigns)
                if hasattr(stmt, 'finalbody'):
                    candidate_assigns += walk_block(stmt.finalbody, local_assigns)
                if hasattr(stmt, 'handlers'):
                    for h in stmt.handlers:
                        if hasattr(h, 'body'):
                            candidate_assigns += walk_block(h.body, local_assigns)

                if has_orelse:
                    local_assigns = []
                else:
                    local_assigns += candidate_assigns

            elif isinstance(stmt, ast.Try):
                candidate_assigns = []
                if hasattr(stmt, 'body'): 
                    candidate_assigns += walk_block(stmt.body, local_assigns)
                if hasattr(stmt, 'orelse'):
                    candidate_assigns += walk_block(stmt.orelse, local_assigns)
                if hasattr(stmt, 'finalbody'):
                    candidate_assigns += walk_block(stmt.finalbody, local_assigns)
                if hasattr(stmt, 'handlers'):
                    for h in stmt.handlers:
                        if hasattr(h, 'body'):
                            candidate_assigns += walk_block(h.body, local_assigns)

                local_assigns += candidate_assigns

            elif isinstance(stmt, (ast.Return)):
                return []

        return local_assigns

    walk_block(tree.body, [])
    return assigns


def get_diff_by_remove(origin, remove):
    diff = []

    for result in origin:
        if result not in remove:
            diff.append(result)

    return diff


def difflib_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def levenstein_similarity(str1, str2):
    return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2))

def get_diff(left, right):
    diff_left = []
    diff_right = []

    for r in right:
        if r not in left:
            diff_right.append(r)

    for l in left:
        if l not in right:
            diff_left.append(l)

    return diff_left, diff_right

def extract_attr_access_var(err):
    msg = err['message']
    rule = err['rule']

    if rule == "reportAttributeAccessIssue":
        pattern = r'"(.*?)"'
        values = re.findall(pattern, msg)

        return values[-1]

    return None

def extract_err_type(err):
    msg = err['message']
    rule = err['rule']

    simple_msg = msg.split("\xa0")[-1]

    if rule == "reportReturnType":
        if simple_msg.startswith("Extra parameter"):
            return {
                "severity": "high"
            }
        else:
            pattern = r'"(.*?)"'
            values = re.findall(pattern, simple_msg)


            try:
                annot = normalize_type(parse_type_str(values[-2]))
                ret_val = normalize_type(parse_type_str(values[-1]))
            except Exception as e:
                return None


            return {
                "severity": "normal",
                "annot": annot,
                "ret": ret_val
            }

    if rule == "reportArgumentType":
        pattern = r'"(.*?)"'
        values = re.findall(pattern, simple_msg)


        try:
            annot = normalize_type(parse_type_str(values[-1]))
            ret_val = normalize_type(parse_type_str(values[-2]))
        except Exception as e:
            return None


        return {
            "severity": "normal",
            "annot": annot,
            "ret": ret_val
        }

    return None

def mismatch_type_num(err):
    info = extract_err_type(err)

    if info is None:
        return -1

    annot = info['annot']
    
    if not annot.args:
        return 1
    return len(annot.args)

def find_variable_or_suggest(variable_dict, var, top_n=3, threshold=0.8):
    if var in variable_dict:
        var_counter = variable_dict[var]
        if top_n == -1:
            return var_counter.most_common()
        return var_counter.most_common(top_n)

    closest_matches = get_close_matches(var, variable_dict.keys(), n=1, cutoff=threshold)

    type_counter = Counter()
    for match in closest_matches:
        # print(match)
        type_counter.update(variable_dict[match])

    type_counter = Counter({k: v for k, v in type_counter.items() if v > 0})

    if top_n == -1:
        top_types = type_counter.most_common()
    else:
        top_types = type_counter.most_common(top_n)
        # top_value = top_types[0][1] if top_types else 0

        # top_types = [(k, v) for k, v in type_counter.items() if v == top_value]
    
    # logger.debug(f"Closest matches for {var}: {top_types}")
    return top_types

def find_top_types(variable_dict, ctx_types, target_ctx_types, var, is_return=False, top_n=-1, threshold=0.8):
    top_types = ()

    if is_return is False:
        type_counter = Counter()
        target_name_infos = target_ctx_types.param_ctx_types[var]

        if var in ctx_types.param_ctx_types:
            candidates = [ctx_types.param_ctx_types[var]]
        else:
            closest_matches = get_close_matches(var, ctx_types.param_ctx_types.keys(), n=1, cutoff=threshold)
            candidates = [ctx_types.param_ctx_types[match] for match in closest_matches]

        for cand_name_infos in candidates:
            for cand_name_info, cand_ctx_info in cand_name_infos.items():
                for cand_ctx, counter in cand_ctx_info.items():
                    for target_name_info, target_ctx_info in target_name_infos.items():
                        for target_ctx in target_ctx_info.keys():
                            if len(cand_ctx.union(target_ctx)) > 0:
                                if len(cand_ctx.intersection(target_ctx)) / len(cand_ctx.union(target_ctx)) >= threshold:
                                    # counter = ctx_type.get(target_ctx, Counter())
                                    type_counter.update(counter)

        top_types = type_counter.most_common()
    else:
        type_counter = Counter()
        target_ctx_candidates = frozenset()
        # if target_ctx_types.ret_ctx_types.values():
        # target_ctx_type = target_ctx_types.ret_ctx_types.values()
            # if target_ctx_type:
            #     target_ctx_candidates = target_ctx_type.values()

        for ctx_name_infos in ctx_types.ret_ctx_types.values():
            for ctx_name_info, ctx_type in ctx_name_infos.items():
                for ctx, counter in ctx_type.items():
                    for target_name_infos in target_ctx_types.ret_ctx_types.values():
                        for target_name_info, target_ctx in target_name_infos.items():
                            if len(ctx.union(target_ctx)) > 0:
                                if len(ctx.intersection(target_ctx)) / len(ctx.union(target_ctx)) >= threshold:
                                    type_counter.update(counter)

        top_types = type_counter.most_common()

    if (not top_types):
        top_types = find_variable_or_suggest(variable_dict, var, top_n=top_n, threshold=threshold)

    if top_n > 0:
        top_types = top_types[:top_n]

    return top_types

def find_top_types_by_tree(variable_dict, ctx_types, target_ctx_types, var, clf, is_return=False, top_n=-1, threshold=0.8):
    top_types = ()

    if is_return is False:
        cand_type_set = set()
        type_rank = []
        target_ctx_name_infos = target_ctx_types.param_ctx_types[var]

        all_datas = []
        all_top_types = []

        for cand_var, cand_name_infos in ctx_types.param_ctx_types.items():
            for cand_name_info, cand_ctx in cand_name_infos.items():
                (cand_filename, cand_class_hierarchy, cand_funcname, cand_args) = cand_name_info

                for target_name_info, target_ctx in target_ctx_name_infos.items():
                    (target_filename, target_class_hierarchy, target_funcname, target_args) = target_name_info
                    datas = make_data(
                        target_ctx,
                        target_filename,
                        target_class_hierarchy,
                        target_funcname,
                        target_ctx_types.decorators,
                        target_args,
                        var,
                        cand_ctx,
                        cand_filename,
                        cand_class_hierarchy,
                        cand_funcname,
                        ctx_types.decorators,
                        cand_args,
                        cand_var
                    )
                    top_types_list = [data[-1] for data in datas]
                    datas = [data[:-1] for data in datas]

                    all_datas.extend(datas)
                    all_top_types.extend(top_types_list)



        all_datas = np.array(all_datas)


        if all_datas.size > 0:

            results = clf.predict_proba(all_datas)
            for idx, result in enumerate(results):
                score = result[1]
                if score > 0.65:
                    for typ in all_top_types[idx]:
                        type_rank.append((typ, score))

    
    else:
        cand_type_set = set()
        type_rank = []
        target_ctx_name_infos = target_ctx_types.ret_ctx_types[var]

        all_datas = []
        all_top_types = []

        for cand_var, cand_name_infos in ctx_types.ret_ctx_types.items():
            for cand_name_info, cand_ctx in cand_name_infos.items():
                (cand_filename, cand_class_hierarchy, cand_funcname, cand_args) = cand_name_info

                for target_name_info, target_ctx in target_ctx_name_infos.items():
                    (target_filename, target_class_hierarchy, target_funcname, target_args) = target_name_info

                    

                    datas = make_data(
                        target_ctx,
                        target_filename,
                        target_class_hierarchy,
                        target_funcname,
                        target_ctx_types.decorators,
                        target_args,
                        var,
                        cand_ctx,
                        cand_filename,
                        cand_class_hierarchy,
                        cand_funcname,
                        ctx_types.decorators,
                        cand_args,
                        cand_var,
                        is_return=True
                    )


                    top_types_list = [data[-1] for data in datas]
                    datas = [data[:-1] for data in datas]

                    for i in range(len(datas)):
                        datas[i][0] = None
                        datas[i][1] = None
                        datas[i][2] = None

                    all_datas.extend(datas)
                    all_top_types.extend(top_types_list)

        all_datas = np.array(all_datas)

        if all_datas.size > 0:
            results = clf.predict_proba(all_datas)
            for idx, result in enumerate(results):
                score = result[1]
                if score > 0.65:
                    for typ in all_top_types[idx]:
                        type_rank.append((typ, score))


    if (not type_rank):
        top_types = find_variable_or_suggest(variable_dict, var, top_n=top_n, threshold=threshold)
    else:
        type_ranking = {} # typ -> num, max_score, total_score
        for typ, score in type_rank:
            info = type_ranking.get(typ, (0,0,0))
            num, max_score, total_score = info
            type_ranking[typ] = (num+1, max(max_score, score), total_score+score)

        top_types = sorted(type_ranking.items(), key=lambda x: ((round(x[1][2] / x[1][0], 4)), x[1][1]), reverse=True)

    if top_n > 0:
        top_types = top_types[:top_n]

    return top_types

def find_type_count(variable_dict, ctx_types, target_ctx_types, var, typ, clf, is_return=False):
    var = str(var)
    top_types = find_top_types_by_tree(variable_dict, ctx_types, target_ctx_types, var, clf, is_return=is_return, top_n=-1, threshold=0.8)
    for top_type, count in top_types:
        if top_type == typ:
            if isinstance(count, int):
                return count
            else:
                return count[0]

    return 0

def fine_type_score(variable_dict, ctx_types, target_ctx_types, var, typ, clf, is_return=False):
    var = str(var)
    top_types = find_top_types_by_tree(variable_dict, ctx_types, target_ctx_types, var, clf, is_return=is_return, top_n=-1, threshold=0.8)
    
    for top_type, score in top_types:
        if top_type == typ:
            if isinstance(score, int):
                total_sum = sum([x[1] for x in top_types])
                return (score / total_sum) if total_sum > 0 else 0
            else:
                return score[1]

    return 0

def check_more_deep(target, cand):
    # if target is more specific than cand, return True
    if target == cand:
        return None

    target_head = str(target.head[0])
    cand_head = str(cand.head[0])

    if target_head == "Any":
        return False
    elif cand_head == "Any":
        return True

    return None
   

@dataclass
class AnalysisResult:
    proj_path: str
    file_path: str
    result_path: str
    func_name: str
    expects: list
    num: int
    analysis_result: list[dict]
    pred: list
    params: list
    param_counter: Counter
    ret_counter: Counter
    is_property: bool
    ctx_types: dict
    target_ctx_types: dict
    total_preds: list
    clf: object
    ret_clf: object

    def is_report_argument_type(self, err):
        return err['rule'] == 'reportArgumentType'

    def is_not_severe(self, err):
        return err['severity'] != 'error' or \
                err['rule'] == 'reportIncompatibleMethodOverride'

    def count_not_severe_error(self):
        return len([err for err in self.analysis_result if self.is_not_severe(err)])

    def count_severe_error(self):
        num = len(self.analysis_result)
        for err in self.analysis_result:
            if err['rule'] == "reportCallIssue":
                num -= 1
                continue

            info = extract_err_type(err)
            if info is not None and info['severity'] == 'normal':
                if str(info['ret']) == "None":
                    # print("ok")
                    num -= 1 

        return num

    def has_undefined_variable(self):
        is_undefined = any(
            err['rule'] == 'reportUndefinedVariable'
            for err in self.analysis_result
        )

        if is_undefined:
            return True

        for err in self.analysis_result:
            if err['rule'] == 'reportAttributeAccessIssue':
                var = extract_attr_access_var(err)
                if var is not None:
                    if var != "None":
                        return True
                

        return False

    @staticmethod
    def naive_compare(x, y):
        x_analysis_result = x.analysis_result
        y_analysis_result = y.analysis_result
        x_err_num = len(x_analysis_result)
        y_err_num = len(y_analysis_result)

        x_has_undefined = x.has_undefined_variable()
        y_has_undefined = y.has_undefined_variable()

        if x_has_undefined and (not y_has_undefined):
            return 1
        elif (not x_has_undefined) and y_has_undefined:
            return -1

        x_severe_num = x.count_severe_error()
        y_severe_num = y.count_severe_error()

        if x_severe_num < y_severe_num:
            return -1
        elif x_severe_num > y_severe_num:
            return 1

        if x_err_num < y_err_num:
            return -1
        elif x_err_num > y_err_num:
            return 1
        else:
            pass

        if x.num < y.num:
            return -1
        elif x.num > y.num:
            return 1
        else:
            return 0


    @staticmethod
    def compare(x, y):
        x_analysis_result = x.analysis_result
        y_analysis_result = y.analysis_result
        # x_analysis_result = [err for err in x.analysis_result if not x.is_report_argument_type(err)]
        # y_analysis_result = [err for err in y.analysis_result if not y.is_report_argument_type(err)]

        if x.num >= 20:
            return -1
        if y.num >= 20:
            return 1

        x_err_num = len(x_analysis_result)
        y_err_num = len(y_analysis_result)

        x_has_undefined = x.has_undefined_variable()
        y_has_undefined = y.has_undefined_variable()

        if STATIC_ANALYSIS:
            if x_has_undefined and (not y_has_undefined):
                return 1
            elif (not x_has_undefined) and y_has_undefined:
                return -1

        x_severe_num = x.count_severe_error()
        y_severe_num = y.count_severe_error()

        if STATIC_ANALYSIS:
            if x_severe_num < y_severe_num:
                return -1
            elif x_severe_num > y_severe_num:
                return 1

        if CONTEXT_SCORE:
            x_total_score = 0.0
            y_total_score = 0.0
            for i, (x_pred, y_pred) in enumerate(zip(x.pred, y.pred)):
                if x_pred is None or y_pred is None:
                    continue

                if str(x_pred) != str(y_pred):
                    if x.params[i] == '__RET__' and not x.is_property:
                            func_name = x.func_name.split(".")[-1]
                            x_score = fine_type_score(x.ret_counter, x.ctx_types, x.target_ctx_types, func_name, str(x_pred), x.ret_clf, is_return=True)
                            y_score = fine_type_score(y.ret_counter, x.ctx_types, x.target_ctx_types, func_name, str(y_pred), x.ret_clf, is_return=True)
                            
                            x_total_score += x_score
                            y_total_score += y_score
                    else:
                        x_score = fine_type_score(x.param_counter, x.ctx_types, x.target_ctx_types, x.params[i], str(x_pred), x.clf)
                        y_score = fine_type_score(y.param_counter, x.ctx_types, x.target_ctx_types, y.params[i], str(y_pred), x.clf)

                        x_total_score += x_score
                        y_total_score += y_score

                        if x_score == 0 and y_score == 0 and x.is_property:
                            x_score = fine_type_score(x.ret_counter, x.ctx_types, x.target_ctx_types, str(x_pred), str(x_pred), x.ret_clf, is_return=True)
                            y_score = fine_type_score(y.ret_counter, x.ctx_types, x.target_ctx_types, str(y_pred), str(y_pred), x.ret_clf, is_return=True)

                            x_total_score += x_score
                            y_total_score += y_score
            
            if x_total_score < y_total_score:
                return 1
            elif x_total_score > y_total_score:
                return -1

        if STATIC_ANALYSIS:

            if x_err_num < y_err_num:
                return -1
            elif x_err_num > y_err_num:
                return 1
            else:
                pass

            if x.num < y.num:
                return -1
            elif x.num > y.num:
                return 1
            else:
                return 0

        return 0
        # return sort_tuple

def revise_any_rank(after_solution_list):
    idx_no_any = -1
    any_idx_list = []

    for idx, sol in enumerate(after_solution_list):
        find_any = False
        for pred in sol.pred:
            if pred is None:
                continue

            if str(pred.head[0]) == "Any":
                find_any = True
                any_idx_list.append(idx)
                break

        if not find_any:
            idx_no_any = idx
            break

    idx_rank = after_solution_list[idx_no_any].num
    high_rank = [after_solution_list[x] for x in any_idx_list if x < idx_rank]
    low_rank = [after_solution_list[x] for x in any_idx_list if x >= idx_rank]

    revise_rank = []

    for after_sol in after_solution_list:
        if after_sol.num == idx_rank:
            revise_rank.extend(high_rank)
            revise_rank.append(after_solution_list[idx_no_any])
            revise_rank.extend(low_rank)
        elif after_sol.num in any_idx_list:
            continue
        else:
            revise_rank.append(after_sol)

    return revise_rank


def naive_rerank(before_solutions, analysis_results, preds, params, param_counter, ret_counter):
    after_solutions = []

    solution_list = []

    for before_sol, analysis_result, pred in zip(before_solutions, analysis_results, preds):
        solution_list.append(AnalysisResult(before_sol, analysis_result, pred, params, param_counter, ret_counter))


    solution_list.sort(key=cmp_to_key(AnalysisResult.naive_compare))
    after_solutions = [sol.num for sol in solution_list]

    return after_solutions

def normalized_shannon_entropy(lst):
    count = Counter(lst)
    total = len(lst)
    unique_count = len(count) 
    
    if unique_count == 1: 
        return 0.0
    
    H = -sum((freq / total) * math.log2(freq / total) for freq in count.values())
    H_max = math.log2(unique_count) 
    
    return H / H_max 

def check_var_change(solution_list, param_counter, ret_counter, funcname, clf, ret_clf):
    minimum_solutions = []
    if not solution_list:
        return solution_list, []

    first_solution = solution_list[0]
    first_report = first_solution.analysis_result
    minimum_len = len(first_report)

    for solution in solution_list:
        if len(solution.analysis_result) == minimum_len:
            minimum_solutions.append(solution)
        else:
            break

    sol_criteria = solution_list

    sol_preds = []
    for sol in sol_criteria:
        sol_preds.append(sol.pred)

    diff_signal_idx_list = []
    max_score = 0.0
    max_idx = -1

    threshold = 0.6
    best_diff_signal_idx_list = []

    if first_solution.total_preds:
        score = normalized_shannon_entropy(first_solution.total_preds)
        if score >= threshold:
            max_score = score
            max_idx = 0
            best_diff_signal_idx_list = [(0, score)]
    else:
        for k, elements in enumerate(zip(*sol_preds)):
            if len(set(elements)) == 1:
                continue

            good_score = round(1 / len(sol_criteria), 3)


            score = normalized_shannon_entropy(elements)
            if score >= threshold:
                diff_signal_idx_list.append((k, score))

            if score == max_score and score >= threshold:
                best_diff_signal_idx_list.append((k, score))
            if score > max_score and score >= threshold:
                max_score = score
                max_idx = k
                best_diff_signal_idx_list = [(k, score)]

    if max_idx == -1:
        return solution_list, []

    # only one change
    diff_signal_idx_list = best_diff_signal_idx_list

    change_type_list = [] # list of list


    if USE_CACHE and os.path.exists(first_solution.result_path / "candidate_types.json"):
        with open(first_solution.result_path / "candidate_types.json", 'r') as f:
            change_type_list = json.load(f)
        
    # check if change_type elements is all empty list
    is_empty = True
    for change_type in change_type_list:
        if change_type:
            is_empty = False
            break

    if (not USE_CACHE) and is_empty:
        change_type_list = []
        for diff_signal_idx, diff_score in diff_signal_idx_list:

            diff_param = str(first_solution.params[diff_signal_idx])
            if diff_param == "__RET__" and not first_solution.is_property:
                suggested_types = find_top_types_by_tree(ret_counter, first_solution.ctx_types, first_solution.target_ctx_types, funcname, ret_clf, is_return=True, top_n=3, threshold=1.0)
 
            else:
                suggested_types = find_top_types_by_tree(param_counter, first_solution.ctx_types, first_solution.target_ctx_types, diff_param, clf, is_return=False, top_n=3, threshold=1.0)

            new_suggested_types = []

            for typ, score in suggested_types:
                pred_type_list = list(zip(*[sol.pred for sol in solution_list]))[diff_signal_idx]

                if str(typ) not in pred_type_list:
                    new_suggested_types.append((typ, score))

            if suggested_types:
                
                signal_expect = first_solution.expects[diff_signal_idx]
                

            change_types = [str(parse_type_str(x[0])) for x in suggested_types]
            change_type_list.append(change_types)


        with open(first_solution.result_path / "candidate_types.json", 'w') as f:
            json.dump(change_type_list, f, indent=4)

    is_empty = True
    for change_type in change_type_list:
        if change_type:
            is_empty = False
            break



    assert len(list(zip(*change_type_list))) <= 3

    non_empty_indices = [i for i, c in enumerate(change_type_list) if c]
    non_empty_count = len(non_empty_indices)

    candidate_combs = []

    if non_empty_count == 1:
        typ_len = len(change_type_list[non_empty_indices[0]])
        for c in change_type_list:
            if c:
                candidate_combs.append(c)
            else:
                candidate_combs.append([None] * typ_len)
    elif non_empty_count > 1:
        for c in change_type_list:
            if c:
                candidate_combs.append([c[0]])
            else:
                candidate_combs.append([None])

    # Run Static Analysis
    candidate_results = []
    new_solutions = []
    tmp_solutions = [sol for sol in solution_list]
    correct_num = []

    run_info_list = []

    try:
        skip_candidates = []
        for i, change_type in enumerate(zip(*candidate_combs)):
            preds = []
            pred_params = {}
            pred_return = None

            idx = 0
            for k, (param, pred) in enumerate(zip(first_solution.params, first_solution.pred)):

                if k in [x[0] for x in diff_signal_idx_list] and change_type[idx] is not None:
                    if param == "__RET__":
                        pred_return = str(change_type[idx])
                    else:
                        pred_params[str(param)] = str(change_type[idx])

                    preds.append(change_type[idx])

                    idx += 1
                else:
                    if param == "__RET__":
                        pred_return = str(pred)
                    else:
                        pred_params[param] = str(pred)

                    preds.append(pred)

            str_expects = [str(x) for x in first_solution.expects]
            str_preds = [str(x) for x in preds]


            check_str_preds = [str(parse_type_str(x).normalized()) for x in str_preds]
            check_str_expects = [str(parse_type_str(x).normalized()) for x in str_expects]



            if check_str_preds == check_str_expects:
                correct_num.append(10+i)

            if not os.path.exists(first_solution.result_path):
                os.makedirs(first_solution.result_path)

            run_info_list.append((
                i,
                first_solution.proj_path,
                first_solution.file_path,
                first_solution.result_path,
                first_solution.func_name,
                pred_params,
                pred_return,
                f"candidate_{i}.json"
            ))


        try:
            shutil.copy(str(first_solution.file_path), str(first_solution.file_path) + '.bak')
            with Pool() as pool:
                pool.map(run_static_analysis_typet5, run_info_list)
        finally:
            if os.path.exists(str(first_solution.file_path) + '.bak'):
                shutil.move(str(first_solution.file_path) + '.bak', str(first_solution.file_path))


        for i, change_type in enumerate(zip(*candidate_combs)):

            with open(first_solution.result_path / f"candidate_{i}.json", 'r') as f:
                analysis_results = json.load(f)['generalDiagnostics']
            
            for mod in analysis_results:
                filename = remove_prefix_filename(mod['file'])
                filename = remove_suffix_filename(filename, is_removed=False)
                mod['file'] = filename

            with open(first_solution.result_path / "removed.json", 'r') as f:
                removed_json = json.load(f)['generalDiagnostics']

            for rem in removed_json:
                filename = remove_prefix_filename(rem['file'])
                filename = remove_suffix_filename(filename, is_removed=True)
                rem['file'] = filename

            analysis_results = get_diff_by_remove(analysis_results, removed_json)

            solution = AnalysisResult(
                first_solution.proj_path,
                first_solution.file_path,
                first_solution.result_path,
                first_solution.func_name,
                first_solution.expects,
                10+i,
                analysis_results,
                preds,
                first_solution.params,
                param_counter,
                ret_counter,
                first_solution.is_property,
                first_solution.ctx_types,
                first_solution.target_ctx_types,
                first_solution.total_preds,
                clf,
                ret_clf
            )

            new_solutions.append(solution)
            tmp_solutions.append(solution)
    except IndexError as e:
        print(e)
        pass

    updated_solution_list = solution_list
    good_sol_list = []


    updated_solution_list = new_solutions + solution_list
    updated_solution_list.sort(key=cmp_to_key(AnalysisResult.compare))


    return updated_solution_list, correct_num
    
def return_rerank(solution_list, file_path, func_name, params, param_counter, ret_counter, clf, ret_clf):
    if params[-1] != "__RET__":
        return solution_list, None
    
    function_body = get_ast_cache(file_path, func_name, params)

    if function_body is None:
        return solution_list, None

    if isinstance(function_body.body[0], (ast.Pass, ast.Ellipsis)):
        return solution_list, None
    if function_body.decorator_list:
        for deco in function_body.decorator_list:
            if isinstance(deco, ast.Name):
                deco_name = deco.id
                if deco_name == "abstractmethod":
                    return solution_list, None

    returns = collect_returns(function_body)

    type_candidate = set([])
    name_candidate = set([])
    call_candidate = set([])
    is_unknown = False


    if len(returns) == 0:
        if isinstance(function_body.body[-1], ast.Raise):
            return solution_list, None
        for node in function_body.body:
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return solution_list, None
        type_candidate.add("None")


    for ret in returns:
        value = ret.value
        if isinstance(value, ast.Constant):
            typ = type(value.value).__name__
            if typ == "NoneType":
                type_candidate.add("None")
            else:
                type_candidate.add(typ)
        elif isinstance(value, ast.Name):
            var = value.id

            if var == "self" or var == "cls":
                func_name_list = func_name.split(".")

                if len(func_name_list) > 1:
                    class_name = func_name_list[-2]
                    type_candidate.add(class_name)
            else:
                
                assigns = collect_assigns_for_return(function_body, var, ret)
                # print(assigns)
                if assigns:
                    for assign in assigns:
                        assign_value = assign.value
                        if isinstance(assign_value, ast.Constant):
                            typ = type(assign_value.value).__name__
                            if typ == "NoneType":
                                type_candidate.add("None")
                            else:
                                type_candidate.add(typ)
                        elif isinstance(assign_value, ast.Name):
                            name_candidate.add(assign_value.id)
                        elif isinstance(assign_value, ast.Call):
                            if isinstance(assign_value.func, ast.Attribute):
                                call_candidate.add(assign_value.func.attr)
                            elif isinstance(assign_value.func, ast.Name):
                                call_candidate.add(assign_value.func.id)
                        else:
                            is_unknown = True
                else:
                    if isinstance(value, ast.Constant):
                        type_candidate.add(str(value))
                    elif isinstance(value, ast.Name):
                        name_candidate.add(value.id)
                    elif isinstance(value, ast.Call):
                        call_candidate.add(value.func.id)
                    else:
                        is_unknown = True
        else:
            is_unknown = True

    if name_candidate:
        for name in name_candidate:
            if name in params:
                type_candidate.add(solution_list[0].pred[params.index(name)])

    if call_candidate:
        for call in call_candidate:
            if call in ["int", "str", "float", "bool", "list", "dict", "set", "tuple"]:
                type_candidate.add(call)
            else:
                if call[0].isupper():
                    type_candidate.add(call)



    ret_types = []
    for sol in solution_list:
        try:
            ret_types.append(parse_type_str(sol.pred[-1]).normalized())
        except Exception as e:
            ret_types.append(None)


    ret_score = [0] * len(solution_list)

    for k, ret_type in enumerate(ret_types):
        if ret_type is None:
            continue
        head_elems = []
        if str(ret_type.head[0]) == "Union":
            head_elems.extend([arg.head[0] for arg in ret_type.args])
            # head_elems.extend([arg.head[0] for arg in ret_type.args])
            # print("OK")
        elif str(ret_type.head[0]) in ["List", "Dict", "Set", "Tuple", "list", "dict", "set", "tuple"]:
            # if "object" == str(elem):
            #     continue
            copy_elem = deepcopy(ret_type)
            copy_elem.args = ()
            head_elems.append(copy_elem)
        else:
            head_elems.append(ret_type.head[0])
        head_elems = [str(head_elem) for head_elem in head_elems]
        ret_type_set = set(head_elems)


        if is_unknown:
            if type_candidate - ret_type_set == set([]):
                ret_score[k] += 1
        else:
            if type_candidate == ret_type_set:
                # print("GOOD")
                ret_score[k] += 2
            elif ret_type_set - type_candidate == set([]):
                ret_score[k] += 1

            

    max_score = max(ret_score)
    # check is only one
    if ret_score.count(max_score) != 1:
        return solution_list, None

    max_idx = ret_score.index(max(ret_score))
    max_ret_type = solution_list[max_idx].pred[-1]            


    first_solution = solution_list[0]
    first_solution_preds = copy(first_solution.pred)

    if max_ret_type == first_solution_preds[-1]:
        return solution_list, None
    else:
        first_solution_preds[-1] = max_ret_type


        check_str_preds = [str(parse_type_str(x).normalized()).lower() for x in first_solution.expects]
        check_str_expects = [str(parse_type_str(x).normalized()).lower() for x in first_solution_preds]

        if check_str_preds == check_str_expects:
            correct_num = 21
        else:
            correct_num = None

        solution = AnalysisResult(
            first_solution.proj_path,
            first_solution.file_path,
            first_solution.result_path,
            first_solution.func_name,
            first_solution.expects,
            21,
            first_solution.analysis_result,
            first_solution_preds,
            first_solution.params,
            param_counter,
            ret_counter,
            first_solution.is_property,
            first_solution.ctx_types,
            first_solution.target_ctx_types,
            first_solution.total_preds,
            clf,
            ret_clf
        )

        return [solution] + solution_list, correct_num


def rerank(proj_path, file_path, result_path, func_name, expects, before_solutions, analysis_results, preds, params, param_counter, ret_counter, is_property, ctx_types, target_ctx_types, total_preds, clf, ret_clf):
    # start_time = time.time()
    
    after_solutions = []

    solution_list = []

    for before_sol, analysis_result, pred in zip(before_solutions, analysis_results, preds):
        solution_list.append(AnalysisResult(
            proj_path,
            file_path,
            result_path,
            func_name,
            expects,
            before_sol, 
            analysis_result, 
            pred, 
            params, 
            param_counter, 
            ret_counter,
            is_property,
            ctx_types,
            target_ctx_types,
            total_preds,
            clf,
            ret_clf
            )
        )

    solution_list.sort(key=cmp_to_key(AnalysisResult.compare))


    correct_num = []
    if not ONLY_STATIC_ANALYSIS:
        
        if STATIC_ANALYSIS:
            solution_list, ret_correct_num = return_rerank(solution_list, file_path, func_name, params, param_counter, ret_counter, clf, ret_clf)
        else:
            ret_correct_num = None
        # print("---")
        if CONTEXT_CANDIDATE:
            solution_list, correct_num = check_var_change(solution_list, param_counter, ret_counter, func_name, clf, ret_clf)


        if ret_correct_num:
            correct_num.append(ret_correct_num)


    if STATIC_CLASSIFICATION:
        new_solution_list = []
        others = []
        for sol in solution_list:
            if len(sol.analysis_result) == 0:
                new_solution_list.append(sol)

        
        after_solutions = new_solution_list


    else:
        after_solutions = solution_list


    return after_solutions, correct_num

