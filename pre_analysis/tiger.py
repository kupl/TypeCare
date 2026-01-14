import json
from pathlib import Path
import libcst as cst
import ast
import re, os, csv
from hityper.typeobject import TypeObject
import shutil
import subprocess
from typet5.type_check import parse_type_str
from multiprocessing import Pool
import time
import json

class TypingImportChecker(ast.NodeVisitor):
    """
    Python ast를 사용하여 'from typing import *' 또는 
    'from typing import Optional'이 있는지 확인하는 Visitor
    """
    def __init__(self):
        self.found_typing_optional = False

    def visit_ImportFrom(self, node):
        # 1. 'from typing' 모듈인지 확인
        if node.module == 'typing':
            # 2. 임포트된 이름 목록 순회 (node.names)
            for alias in node.names:
                
                # 'import *' (ImportStar) 확인
                if alias.name == '*':
                    self.found_typing_optional = True
                    return
                
                # 'import Optional' 확인 (as Optional은 Optional로 간주)
                elif alias.name == 'Optional' or alias.asname == 'Optional':
                    self.found_typing_optional = True
                    return
        
        # 순회를 계속합니다.
        self.generic_visit(node)

def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return None
        else:
            res = second_matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
            if (" " in res and "[" not in res) or res.lower() == "unknown":
                res = None
            return res
    else:
        res = matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
        if (" " in res and "[" not in res) or res.lower() == "unknown":
            res = None
        return res

def match_type(string):
    string = string.split("\nPython Code:")[0].split("\nQ:")[0]
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return string.split("\n")[0][:-1]
        else:
            return second_matched[0].replace("`", "")
    else:
        return matched[0].replace("`", "")

def match_type_for_completion(string):
    string = string.split("\nPython Code:")[0].split("\nQ:")[0]
    pattern = re.compile(r'[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'[a-zA-Z\.\,\[\] ]+')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return string.split("\n")[0][:-1]
        else:
            return second_matched[0].replace("`", "")
    else:
        return matched[0].replace("`", "")


def extract_type_from_text(text):
    if len(text.split()) > 0:
        text = text.split()[0]
    else:
        text = text
    if text.endswith(".") or text.endswith(","):
        text = text[:-1]
    typeobjs = TypeObject.Str2Obj(text)
    return typeobjs


def extract_type_from_cot(text):
    text = text.split()[-1][:-1]
    typeobjs = TypeObject.Str2Obj(text)
    return typeobjs

def transform_sample_to_top(data, cot = False, case_sensitive = True):
    freq = {}
    for d in data:
        if cot:
            d = match_type_for_cot(d)
            if d == None:
                continue
        else:
            d = match_type(d)
            if d == None:
                continue
        found = None
        for k in freq:
            if k.lower() == d.lower() and not case_sensitive:
                found = k
                break
            elif k == d and case_sensitive:
                found = k
                break
        if found != None:
            freq[found] += 1
        else:
            freq[d] = 1

    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top = []
    for s in sorted_freq:
        top.append(s[0])
    return top

class FunctionLocator(ast.NodeVisitor):
    def __init__(self):
        self.inclass = False
        self.inclass = False
        self.found = False
        self.node = None


    def visit_ClassDef(self, node):
        if not self.inclass and node.name == self.classname:
            self.inclass = True
            self.found = False
            self.generic_visit(node)
            if self.found and self.funcname == "global":
                self.node = node
        elif not self.inclass:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        if not self.infunc and node.name == self.funcname and self.inclass:
            if self.scope == 'return' and node.name == self.name:
                self.node = node
            else:
                self.infunc = True
                self.found = False
                self.generic_visit(node)
                if self.found:
                    self.node = node
        elif not self.infunc and self.inclass:
            self.generic_visit(node)
    
    def visit_Name(self, node):
        if node.id == self.name and self.scope == 'local' and self.infunc and self.inclass:
            self.found = True
    
    def visit_Attribute(self, node):
        if node.attr == self.name and hasattr(node.value, "id") and node.value.id == "self" and self.scope == "local" and self.infunc and self.inclass:
            self.found = True

    def visit_arg(self, node):
        if node.arg == self.name and self.scope == 'arg' and self.infunc and self.inclass:
            self.found = True


    def run(self, root, loc, name, scope):
        self.inclass = False
        self.infunc = False
        self.node = None
        self.found = False
        self.funcname, self.classname = loc.split('@')
        self.name = name
        self.scope = scope
        if self.classname == 'global':
            self.inclass = True
        if self.funcname == 'global':
            self.infunc = True
        if self.inclass and self.infunc:
            remover = GlobalNodeRemover()
            node = remover.run(root)
            return node
        else:
            self.visit(root)
        return self.node

def annotate_function(func_node: ast.FunctionDef, typ: str, var_name, is_param=True, is_removed=False) -> ast.FunctionDef:
    if is_removed:
        if is_param:
            for arg in func_node.args.posonlyargs:
                if arg.arg == var_name:
                    arg.annotation = None
                    break
            for arg in func_node.args.args:
                if arg.arg == var_name:
                    arg.annotation = None
                    break
            if func_node.args.vararg and func_node.args.vararg.arg == var_name:
                func_node.args.vararg.annotation = None
            elif func_node.args.kwarg and func_node.args.kwarg.arg == var_name:
                func_node.args.kwarg.annotation = None
            elif func_node.args.kwonlyargs:
                for arg in func_node.args.kwonlyargs:
                    if arg.arg == var_name:
                        arg.annotation = None
                        break

        else:
            func_node.returns = None
    else:
        if is_param:
            for arg in func_node.args.posonlyargs:
                if arg.arg == var_name:
                    arg.annotation = ast.Name(id=typ, ctx=ast.Load())
                    break
            for arg in func_node.args.args:
                if arg.arg == var_name:
                    arg.annotation = ast.Name(id=typ, ctx=ast.Load())
                    break
            if func_node.args.vararg and func_node.args.vararg.arg == var_name:
                func_node.args.vararg.annotation = ast.Name(id=typ, ctx=ast.Load())
            elif func_node.args.kwarg and func_node.args.kwarg.arg == var_name:
                func_node.args.kwarg.annotation = ast.Name(id=typ, ctx=ast.Load())
            elif func_node.args.kwonlyargs:
                for arg in func_node.args.kwonlyargs:
                    if arg.arg == var_name:
                        arg.annotation = ast.Name(id=typ, ctx=ast.Load())
                        break
        else:
            func_node.returns = ast.Name(id=typ, ctx=ast.Load())
    
    return func_node

def process_prediction(args):
    index, pred, tiger_path, proj_path, result_path, file_path, class_name, method_name, var_name, kind = args

    if index is None:
        # Handle the case for removed code

        # analysis removed code
        try:
            copy_path = f"{file_path[:-3]}_removed.py"
            shutil.copyfile(file_path, copy_path)

            with open(copy_path, "r") as f:
                code = f.read()

            root = ast.parse(code)
            locator = FunctionLocator()
            if kind == "ret":
                kind = "return"
            node = locator.run(root, f"{method_name}@{class_name}", var_name, kind)

            if node is None:
                return

            func_node = annotate_function(node, None, var_name, kind == "arg", True)


            with open(copy_path, "w") as f:
                f.write(ast.unparse(root))

            p = subprocess.Popen(
                ['pyright', '--outputjson', copy_path],
                stdout=subprocess.PIPE,
                cwd=tiger_path / proj_path
            )
            output, err = p.communicate()
            output_json = output.decode("utf-8")

            with open(result_path / f"removed.json", 'w') as f:
                f.write(output_json)
        finally:
            if os.path.exists(copy_path):
                os.remove(copy_path)
    else:
        try:
            parse_predtype = parse_type_str(str(pred))
        except:
            return

        copy_path = f"{file_path[:-3]}_{index}.py"

        try:
            shutil.copyfile(file_path, copy_path)

            with open(copy_path, "r") as f:
                code = f.read()

            root = ast.parse(code)
            locator = FunctionLocator()
            if kind == "ret":
                kind = "return"
            node = locator.run(root, f"{method_name}@{class_name}", var_name, kind)

            if node is None:
                return

            annotate_function(node, str(parse_predtype), var_name, kind == "arg")

            with open(copy_path, "w") as f:
                f.write(ast.unparse(root))

            p = subprocess.Popen(
                ['pyright', '--outputjson', copy_path],
                stdout=subprocess.PIPE,
                cwd=tiger_path / proj_path
            )
            output, err = p.communicate()
            output_json = output.decode("utf-8")

            with open(result_path / f"modified_{index}.json", 'w') as f:
                f.write(output_json)
        finally:
            if os.path.exists(copy_path):
                os.remove(copy_path)



def run():
    tiger_path = Path.home()
    pred_path = Path("prediction") / "Tiger"

    with open(pred_path / "transformed_result.json", "r") as f:
        tiger_result = json.load(f)

    file_time_dict = {}
    start_time = time.time()

    for test_info in tiger_result:
        file_path = test_info['file_path']
        proj_path = test_info['src_path']

        try:
            result_path = Path(test_info['result_path'])

            save_path = Path('data') / 'Tiger_test' / result_path

            print(f"Processing {proj_path}")

            if not save_path.exists():
                os.makedirs(save_path)

            file_start_time = time.time()
            
            predictions = test_info['predictions']
            expects = test_info['expects']
        
            correct_idx = []
            incorrect_idx = []

            for index, pred in enumerate(predictions):
                try:
                    parse_predtype = parse_type_str(pred[0]).normalized()
                    parse_gttype = parse_type_str(expects[0]).normalized()

                    if parse_predtype == parse_gttype:
                        correct_idx.append(index)
                    else:
                        incorrect_idx.append(index)
                except:
                    incorrect_idx.append(index)

            info = {
                'sig_name': str(result_path).replace('+', '/'),
                'correct': correct_idx,
                'incorrect': incorrect_idx,
            }


            with open(save_path / "info.json", "w") as f:
                json.dump(info, f, indent=4)

            info_str = str(result_path).split('+')[-1]

            contexts = str(info_str).split('--')[1]
            var_name = str(info_str).split('--')[2]

            file_info = {
                'file_path' : file_path,
                'sig_name' : contexts,
                'var_name' : var_name,
            }


            with open(save_path / "file_info.json", "w") as f:
                json.dump(file_info, f, indent=4)

            with open(tiger_path / file_path, "r") as f:
                original_code = ast.parse(f.read())

            import_checker = TypingImportChecker()
            import_checker.visit(original_code)
            if import_checker.found_typing_optional:
                new_predictions = []
                for preds in predictions:
                    new_preds = []
                    for pred in preds:
                        if pred is None:
                            new_preds.append(pred)
                            continue
                        new_pred = str(pred).replace('typing.Optional', 'Optional')
                        new_preds.append(new_pred)
                    new_predictions.append(new_preds)
                if predictions != new_predictions:
                    print(f'Normalized predictions for Optional in {file_path}')
                    print(f'Before: {predictions}')
                    print(f'After: {new_predictions}')
                predictions = new_predictions

            class_name = contexts.split('@')[1]
            method_name = contexts.split('@')[0]

            if test_info['params'][0] == "__RET__":
                kind = "ret"
            else:
                kind = "arg"

            file_path = str(tiger_path / file_path)

            try:
                shutil.copyfile(file_path, file_path + ".bak")
                # make data
                datas = []
                for index, pred in enumerate(predictions):
                    datas.append(
                        (index, pred[0], tiger_path, proj_path, save_path, file_path, class_name, method_name, var_name, kind)
                    )

                # add for removed code
                datas.append(
                    (None, expects[0], tiger_path, proj_path, save_path, file_path, class_name, method_name, var_name, kind)
                )

                # Use multiprocessing to process predictions
                with Pool() as pool:
                    pool.map(process_prediction, datas)
            finally:
                # restore original file
                if os.path.exists(file_path + ".bak"):
                    shutil.move(file_path + ".bak", file_path)


            file_end_time = time.time()
            file_time_dict[str(save_path)] = round(file_end_time - file_start_time, 2)


        except FileNotFoundError as e:
            print(f"File not found: {file_path}. Error: {e}")
            continue

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    print(f"Processed {len(tiger_result)} test cases.")

    with open('time_dict.json', 'w') as f:
        json.dump(file_time_dict, f, indent=4)


if __name__ == "__main__":
    run()