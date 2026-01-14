import pickle
from pathlib import Path

from typet5.static_analysis import FunctionSignature
import libcst as cst
import ast
from collections import Counter
import shutil
import os
import subprocess
from typet5.type_check import parse_type_expr, parse_type_str
import json
from multiprocessing import Pool
import time

from pprint import pprint
from config import TYPET5_PREDICTION_PATH

class TypingOptionalCheckerLegacy(cst.CSTVisitor):
    """
    libcst 헬퍼 함수 없이 from typing import Optional 구문을 확인하는 Visitor
    """
    def __init__(self):
        super().__init__()
        self.found_optional_import: bool = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # from typing ... 인지 확인
        # node.module이 None이 아니고 (상대 경로가 아닌 경우), cst.Name 노드이며, 이름이 'typing'인지 확인
        module_is_typing = (
            node.module is not None and
            isinstance(node.module, cst.Name) and
            node.module.value == "typing"
        )

        if module_is_typing:
            # import Optional 이 있는지 확인
            if isinstance(node.names, cst.ImportStar):
                self.found_optional_import = True
                return
            
            for import_alias in node.names:
                # import_alias.name이 cst.Name 노드이며, 이름이 'Optional'인지 확인
                is_optional_imported = (
                    isinstance(import_alias.name, cst.Name) and
                    import_alias.name.value == "Optional"
                )

                if is_optional_imported:
                    self.found_optional_import = True
                    return

# Get annotation number of the method (or function) signature from the given code
def count_annotations(code: str, target: str):
    tree = ast.parse(code)
    annotation_counter = Counter()

    def normalize_annotation(annotation):
        return str(normalize_type(parse_type_str(annotation)))

    find_target = False

    def recursive_count_annotations(node, parent=None):
        current_node = StackNode(node.name, parent) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else parent

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and current_node.get_full_name() == target:
            nonlocal find_target
            find_target = True
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                return_type = normalize_annotation(ast.unparse(node.returns))
                annotation_counter[return_type] += 1
            for arg in node.args.args:
                if arg.annotation:
                    annotation_type = normalize_annotation(ast.unparse(arg.annotation))
                    annotation_counter[annotation_type] += 1

        for child in ast.iter_child_nodes(node):
            recursive_count_annotations(child, current_node)



    recursive_count_annotations(tree)
    assert find_target, f"Target method {target} not found in the code"
    

    return annotation_counter


class OverrideAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,) 

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):

        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:

        current_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        check_class_pos = len(self.target_class) - 1

        while current_node:
            if isinstance(current_node, cst.ClassDef):
                if current_node.name.value == self.target_class[check_class_pos]:
                    if check_class_pos == 0:
                        return True
                    else:
                        check_class_pos -= 1
                else:
                    return False
            elif isinstance(current_node, cst.Module):
                return False
            current_node = self.get_metadata(cst.metadata.ParentNodeProvider, current_node)
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:

        if self.target_class:

            if not self._is_in_target_class(original_node):
                return updated_node


        if original_node.name.value == self.target_function:

            new_params = []
            for param in updated_node.params.params:
                param_name = param.name.value
                if param_name in self.annotations:

                    new_annotation = self.annotations[param_name]
                    new_param = param.with_changes(annotation=new_annotation)
                else:
                    new_param = param.with_changes(annotation=None)
                new_params.append(new_param)

            new_return_annotation = self.return_annotation

            return updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )
        return updated_node

class RemoveAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:
        current_node = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        check_class_pos = len(self.target_class) - 1

        while current_node:
            if isinstance(current_node, cst.ClassDef):
                if current_node.name.value == self.target_class[check_class_pos]:
                    if check_class_pos == 0:
                        return True
                    else:
                        check_class_pos -= 1
                else:
                    return False
            elif isinstance(current_node, cst.Module):
                return False
            current_node = self.get_metadata(cst.metadata.ParentNodeProvider, current_node)
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if self.target_class:
            if not self._is_in_target_class(original_node):
                return updated_node

        if original_node.name.value == self.target_function:
            # 각 파라미터 주석 제거
            new_params = []
            for param in updated_node.params.params:
                # param name
                param_name = param.name.value
                if param_name in self.annotations:
                    new_param = param.with_changes(annotation=None)
                else:
                    new_param = param
                new_params.append(new_param)

            new_return_annotation = None

            return updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )
        return updated_node


class VarAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(self, target_name, original_annot, target_annot, in_class):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_var = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_var = target_name
        self.original_annot = original_annot
        self.target_annot = target_annot
        self.in_class = in_class

    def leave_AnnAssign(self, original_node, updated_node):
        if updated_node.target.value == self.target_var and self._has_original_annotation(updated_node.annotation):
            return updated_node.with_changes(annotation=self.target_annot)
        elif self.in_class:
            if isinstance(updated_node.target, cst.Attribute):
                temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=updated_node.target)])])
                if temp_module.code.startswith("self."):
                    if updated_node.target.attr == self.target_var and self._has_original_annotation(updated_node.annotation):
                        return updated_node.with_changes(annotation=self.target_annot)

        return updated_node

    def _has_original_annotation(self, annotation):
        return parse_type_expr(annotation.annotation) == parse_type_expr(self.original_annot.annotation)

class RemoveVarAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,) 

    def __init__(self, target_name, original_annot, in_class):
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_var = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_var = target_name
        self.original_annot = original_annot
        self.in_class = in_class


    def leave_AnnAssign(self, original_node, updated_node):
        if updated_node.target.value == self.target_var and self._has_original_annotation(updated_node.annotation):
            if updated_node.value is None:
                return cst.SimpleStatementLine(
                    body=[cst.Expr(value=updated_node.target)]
                )
            # Otherwise, transform normally
            return cst.Assign(
                targets=[cst.AssignTarget(target=updated_node.target)],
                value=updated_node.value,
            )
        elif self.in_class:
            if isinstance(updated_node.target, cst.Attribute) and parse_type_expr(updated_node.target.value).startswith("self."):
                if updated_node.target.attr == self.target_var and self._has_original_annotation(updated_node.annotation):
                    if updated_node.value is None:
                        return cst.SimpleStatementLine(
                            body=[cst.Expr(value=updated_node.target)]
                        )
                    # Otherwise, transform normally
                    return cst.Assign(
                        targets=[cst.AssignTarget(target=updated_node.target)],
                        value=updated_node.value,
                    )

        return updated_node
    
    def _has_original_annotation(self, annotation):
        return parse_type_expr(annotation.annotation) == parse_type_expr(self.original_annot.annotation)

def make_annotation(params, ret, preds):
    annotations = {}
    for i, param in enumerate(params):
        new_type = cst.parse_expression(str(preds[i]))
        annotations[param] = cst.Annotation(new_type)

    return annotations, cst.Annotation(cst.Name(preds[-1]))

def annotations_equal_by_code(annot1, annot2):
    return cst.Module([]).code_for_node(annot1.annotation) == cst.Module([]).code_for_node(annot2.annotation)

def annotation_to_str(annotation):
    return cst.Module([]).code_for_node(annotation.annotation)

def normalize_annotation(annotation):
    return str((parse_type_str(annotation)).normalized())

def process_prediction(args):
    index, candidate_preds, ret_idx, predicted_params, target_param_indexes, proj_path, result_path, file_path, sig_name, wrapper = args

    if index is None:
        # Handle the case for removed code

        # analysis removed code
        try:
            copy_path = f"{file_path[:-3]}_removed.py"
            shutil.copyfile(file_path, copy_path)

            preds = candidate_preds

            if preds[ret_idx] is None:
                pred_returns = None
            else:
                pred_returns = cst.Annotation(cst.parse_expression(preds[ret_idx]))

            pred_params = {
                param: cst.Annotation(cst.parse_expression(preds[i])) for i, param in enumerate(predicted_params) if i in target_param_indexes
            }

            remove_transformer = RemoveAnnotationTransformer(sig_name, pred_params, pred_returns)
            remove_module = wrapper.visit(remove_transformer)

            with open(copy_path, "w") as f:
                f.write(remove_module.code)

            p = subprocess.Popen(
                ['pyright', '--outputjson', str(copy_path)],
                stdout=subprocess.PIPE,
                cwd=str(proj_path)
            )
            output, err = p.communicate()
            output_json = output.decode("utf-8")

            with open(result_path / f"removed.json", 'w') as f:
                f.write(output_json)
        finally:
            if os.path.exists(copy_path):
                os.remove(copy_path)
    else:
        copy_path = f"{file_path[:-3]}_{index}.py"

        try:
            shutil.copyfile(file_path, copy_path)

            preds = candidate_preds
            if preds[ret_idx] is None:
                pred_returns = None
            else:
                pred_returns = cst.Annotation(cst.parse_expression(preds[ret_idx]))

            pred_params = {
                param: cst.Annotation(cst.parse_expression(preds[i])) for i, param in enumerate(predicted_params) if i in target_param_indexes
            }

            override_transformer = OverrideAnnotationTransformer(sig_name, pred_params, pred_returns)
            override_module = wrapper.visit(override_transformer)

            with open(copy_path, "w") as f:
                f.write(override_module.code)

            p = subprocess.Popen(
                ['pyright', '--outputjson', str(copy_path)],
                stdout=subprocess.PIPE,
                cwd=str(proj_path)
            )
            output, err = p.communicate()
            output_json = output.decode("utf-8")

            with open(result_path / f"modified_{index}.json", 'w') as f:
                f.write(output_json)
        finally:
            if os.path.exists(copy_path):
                os.remove(copy_path)


def run():
    result_path = Path('prediction/TypeT5')

    with open(result_path / 'typet5_function_transform.pkl', 'rb') as f:
        evalr = pickle.load(f)

    # prj_roots = evalr.project_roots
    # print(f'Number of project roots: {len(prj_roots)}')


    # result = evalr.return_predictions()
    result_path_set = set()

    path_set = set([])

    start_time = time.time()
    sig_count = 0
    file_time_dict = {}


    # for data in evalr:
    #     expected = data['expects']
    #     predictions = data['predictions']
    #     params = data['params']

    #     print(params)
    #     print(predictions)

    #     if not params:
    #         continue

    #     repo_name = data['repo_name']
    #     file_path = data['file_path']
    #     proj_path = file_path.split(repo_name)[0] + repo_name
    #     sig_name = data['target']

    #     # check if file exists
    #     if not os.path.exists(file_path):
    #         print(f'File not found: {file_path}')
    #         continue

    #     file_time_start = time.time()

    #     result_path = Path('data/TypeT5_11054') / data['result_path']

    #     correct_set = set()
    #     incorrect_set = set()

    #     if '__RET__' in params:
    #         ret_idx = params.index('__RET__')
    #         target_param_indexes = [i for i in range(len(params)) if i != ret_idx]
    #     else:
    #         print(f'Return type not found in params for {file_path}')
    #         raise Exception()
        
    #     # match type
    #     for i, preds in enumerate(predictions):
    #         is_real_match = True
    #         for pred, expect in zip(preds, expected):
    #             if pred is None:
    #                 if expect is None:
    #                     continue
    #                 else:
    #                     raise Exception("Prediction is None but expected is not None")

    #             if normalize_annotation(pred) != normalize_annotation(expect):
    #                 is_real_match = False
    #                 break

    #         if is_real_match:
    #             correct_set.add(i)
    #         else:
    #             incorrect_set.add(i)

    #     with open(str(file_path), 'r') as f:
    #         code = f.read()

    #     module = cst.parse_module(code)
    #     wrapper = cst.metadata.MetadataWrapper(module)

    #     checker_optional = TypingOptionalCheckerLegacy()
    #     wrapper.visit(checker_optional)
    #     uses_optional = checker_optional.found_optional_import
        
    #     if uses_optional:
    #         # change typing.Optional to Optional in predictions
    #         new_predictions = []
    #         for preds in predictions:
    #             new_preds = []
    #             for pred in preds:
    #                 if pred is None:
    #                     new_preds.append(pred)
    #                     continue
    #                 new_pred = str(pred).replace('typing.Optional', 'Optional')
    #                 new_preds.append(new_pred)
    #             new_predictions.append(new_preds)
    #         if predictions != new_predictions:
    #             print(f'Normalized predictions for Optional in {file_path}')
    #             print(f'Before: {predictions}')
    #             print(f'After: {new_predictions}')
    #         predictions = new_predictions


    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)

    #     info = {
    #         'sig_name': data['target'],
    #         'correct': list(correct_set),
    #         'incorrect': list(incorrect_set),
    #     }


    #     with open(result_path / "info.json", 'w') as f:
    #         json.dump(info, f)


    #     print(f'Processing {file_path}...')

    #     try:
    #         shutil.copy(str(file_path), str(file_path) + '.bak')
    #         datas = []

    #         for num, candidate_preds in enumerate(predictions):
    #             datas.append(
    #                 (num, candidate_preds, ret_idx, params, target_param_indexes, str(proj_path), result_path, str(file_path), sig_name, wrapper)
    #             )
            
    #         datas.append(
    #             (None, predictions[0], ret_idx, params, target_param_indexes, str(proj_path), result_path, str(file_path), sig_name, wrapper)
    #         )
    #         with Pool() as pool:
    #             pool.map(process_prediction, datas)
    #     finally:
    #         # restore original file
    #         if os.path.exists(str(file_path) + ".bak"):
    #             shutil.move(str(file_path) + ".bak", str(file_path))

    #     file_time_end = time.time()
    #     file_time = file_time_end - file_time_start
    #     file_time_dict[str(result_path)] = file_time

    
    with open(TYPET5_PREDICTION_PATH, 'rb') as f:
        evalr = pickle.load(f)

    prj_roots = evalr.project_roots
    # print(f'Number of project roots: {len(prj_roots)}')


    result = evalr.return_predictions()
    total_datas = []
    for proj, sig_map in result.items():
        print(f'Project: {proj}')
        proj_name = proj.name

        for path, sig in sig_map.items():
            print(f'Path: {path}')
            # print(f'Signature: {sig}')

            pid = evalr.find_projects(proj_name)[0]
            expected = evalr.label_maps[pid][path]
            top_pred = evalr.predictions[pid].elem2preds[path]
            all_predcitions = evalr.predictions[pid].elem2all_preds[path]
            # print(f'Expected: {expected}')

            if isinstance(sig, FunctionSignature):
                expected_params = expected.params
                expected_returns = expected.returns
                predicted_params = sig.params
                predicted_returns = sig.returns
                try:
                    predicted_returns = str(parse_type_str(cst.Module([]).code_for_node(predicted_returns.annotation)).normalized())
                except:
                    continue

                # check expected_params is all None
                is_all_none = True
                for param in expected_params.values():
                    if param != None:
                        is_all_none = False
                        break

                if (is_all_none or (not expected_params)) and expected_returns == None:
                    continue

                sig_count += 1

                sig_name = str(path).split('/')[1]
                src_name = str(path).split('/')[0]
                src_path = src_name.replace('.', '/') + '.py'

                first_dir = src_name.split('.')[0]

                proj_path = Path(str(proj).replace("/home/wonseokoh/TypeT5/ManyTypes4Py/repos/test", str(Path.home() / "TypeCare" / "BetterTypes4Py" / "repos" / "test")))

                file_path = proj_path / src_path
                # check if file exists
                if not os.path.exists(file_path):
                    file_path = proj_path / 'src' / src_path

                    # check if file exists
                    if not os.path.exists(file_path):
                        print(f'File not found: {file_path}')
                        continue 

                file_time_start = time.time()

                file_name = Path(str(proj_name)) / src_name.replace('.', '_')
                result_path = Path('data/TypeT5_1118') / str(file_name).replace('/', '_') / sig_name.replace('.', '_')


                # if str(result_path) != "data/TypeT5_1118/ShadowTemplate__beautiful-python-3/design_patterns_behavioural_command_mixer/MuteVolumeCommand___init__":
                #     continue

                # Check Parameter
                param_infos = list(expected_params.items())
                predicted_params_keys = list(predicted_params.keys())

                target_param_indexes = []

                for (param_key, typ) in param_infos:
                    if typ is None:
                        continue
                    if param_key in predicted_params_keys:
                        target_param_indexes.append(predicted_params_keys.index(param_key))

                
                top_pred = [str(pred.normalized()) for pred in top_pred]
                ret_idx = top_pred.index(predicted_returns)
                assert ret_idx >= 0, "Return type not found in predictions"


                expected_types = []

                for param, expected_type in expected_params.items():
                    if expected_type == None:
                        continue

                    code = cst.Module([]).code_for_node(expected_type.annotation)
                    expected_type = str(parse_type_str(code).normalized())
                    expected_types.append(expected_type)


                correct_set = set()
                incorrect_set = set()

                # print(proj_name)
                # print(pid)
                # print(sig_name)
                # print(src_path)
                # # print(file_path)
                # print(expected_params)
                # print(predicted_params)
                # print(target_param_indexes)
                # print(all_predcitions)


                expects = []

                for param, expected_type in expected_params.items():
                    if expected_type == None:
                        expects.append(None)
                        continue

                    expected_type = str(parse_type_expr(expected_type.annotation).normalized())
                    expects.append(expected_type)

                if expected_returns == None:
                    expects.append(None)
                else:
                    code = cst.Module([]).code_for_node(expected_returns.annotation)
                    expected_type = str(parse_type_str(code).normalized())
                    expects.append(expected_type)

                # if len(all_predcitions[0]) != len(expects):
                #     # assert ret_idx == (len(expects) - 1), print(all_predcitions[0], expects, ret_idx)
                #     new_all_predictions = []
                #     for predictions in all_predcitions:
                #         new_all_predictions.append(predictions[:len(expects)])

                #     all_predcitions = new_all_predictions

                #     t1 = [str(t.normalized()) for t in all_predcitions[0]]

                #     test_t = []
                #     for t in predicted_params.values():
                #         code = cst.Module([]).code_for_node(t.annotation)
                #         t = str(parse_type_str(code).normalized())
                #         test_t.append(t)

                #     t2 = test_t + [predicted_returns]

                #     assert t1 == t2, print(t1, t2, expects) 

                # data = {
                #     "repo_name": proj_name,
                #     "file_path": src_path,
                #     "target": sig_name,
                #     "params": predicted_params_keys + ["__RET__"],
                #     "predictions": [[str(t) for t in predictions ] for predictions in all_predcitions],
                #     "expects": expects
                # }

                # total_datas.append(data)

                # continue

                
                # match type
                for i, preds in enumerate(all_predcitions):
                    preds = [str(pred.normalized()) for pred in preds]
                    returns = preds[ret_idx]
                    preds = [preds[i] for i in target_param_indexes]   

                    assert len(preds) == len(expected_types)

                    is_real_match = True
                    for pred, expect in zip(preds, expected_types):
                        assert expect is not None

                        if str(pred) != str(expect):
                            is_real_match = False
                            break

                    if is_real_match:
                        if expected_returns != None:
                            # returns = preds[-1]
                            if str(returns) != str(parse_type_expr(expected_returns.annotation).normalized()):
                                is_real_match = False
                                # break

                    if is_real_match:
                        correct_set.add(i)
                    else:
                        incorrect_set.add(i)

                with open(str(file_path), 'r') as f:
                    code = f.read()

                module = cst.parse_module(code)
                wrapper = cst.metadata.MetadataWrapper(module)
                
                new_all_predictions = []
                for preds in all_predcitions:
                    preds = [str(pred) for pred in preds]
                    new_all_predictions.append(preds)
                all_predcitions = new_all_predictions

                checker_optional = TypingOptionalCheckerLegacy()
                wrapper.visit(checker_optional)
                uses_optional = checker_optional.found_optional_import
                
                if uses_optional:
                    # change typing.Optional to Optional in predictions
                    new_predictions = []
                    for preds in all_predcitions:
                        new_preds = []
                        for pred in preds:
                            if pred is None:
                                new_preds.append(pred)
                                continue
                            new_pred = str(pred).replace('typing.Optional', 'Optional')
                            new_preds.append(new_pred)
                        new_predictions.append(new_preds)
                    if all_predcitions != new_predictions:
                        print(f'Normalized predictions for Optional in {file_path}')
                        print(f'Before: {all_predcitions}')
                        print(f'After: {new_predictions}')
                    all_predcitions = new_predictions

                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                info = {
                    'sig_name': sig_name,
                    'correct': list(correct_set),
                    'incorrect': list(incorrect_set),
                }


                with open(result_path / "info.json", 'w') as f:
                    json.dump(info, f)


                try:
                    shutil.copy(str(file_path), str(file_path) + '.bak')
                    datas = []

                    for num, candidate_preds in enumerate(all_predcitions):
                        datas.append(
                            (num, candidate_preds, ret_idx, predicted_params_keys, target_param_indexes, str(proj_path), result_path, str(file_path), sig_name, wrapper)
                        )
                    
                    datas.append(
                        (None, all_predcitions[0], ret_idx, predicted_params_keys, target_param_indexes, str(proj_path), result_path, str(file_path), sig_name, wrapper)
                    )
                    with Pool() as pool:
                        pool.map(process_prediction, datas)
                finally:
                    # restore original file
                    if os.path.exists(str(file_path) + ".bak"):
                        shutil.move(str(file_path) + ".bak", str(file_path))

                file_time_end = time.time()
                file_time = file_time_end - file_time_start
                file_time_dict[str(result_path)] = file_time

    with open('typet5_transformed_result.json', 'w') as f:
        json.dump(total_datas, f, indent=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Total signatures processed: {sig_count}")

    with open('file_time_dict.json', 'w') as f:
        json.dump(file_time_dict, f, indent=4)


if __name__ == '__main__':
    run()