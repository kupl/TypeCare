from pathlib import Path
import json
import subprocess
import libcst as cst
from multiprocessing import Pool
import shutil
import os
import ast

from .utils import TypingOptionalCheckerLegacy, CSTAnnotator
from .utils import FunctionLocator, annotate_function
from typet5.type_check import parse_type_str
from argparse import ArgumentParser

class PreAnalysis:
    tool_name: str

    def load_data(self):
        model_output_path = Path("prediction") / self.tool_name

        with open (model_output_path / "transformed_result.json", "r") as f:
            data = json.load(f)

        return data
    
    def run_pyright(self, project_path, file_path):
        p = subprocess.Popen(
            ['pyright', '--outputjson', file_path],
                stdout=subprocess.PIPE,
                cwd=project_path
        )
        output, _ = p.communicate()
        output_json = output.decode('utf-8')

        return output_json

    def check_optional(self, code):
        module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)

        checker_optional = TypingOptionalCheckerLegacy()
        wrapper.visit(checker_optional)
        uses_optional = checker_optional.found_optional_import

        return uses_optional
    
    def change_optional_type(self, predictions):
        # Change 'typing.Optional' to 'Optional' in predictions
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

        return new_predictions

    def process_single(self, data):
        i, predictions, project_path, file_path, relative_path, params, output_path = data

        annotated_code = self.annotate(predictions, is_removed=(i == None))
        if annotated_code is None:
            return
        
        if i is not None:
            json_name = f"modified_{i}.json"
            copy_path = f"{str(file_path)[:-3]}_{i}.py"
            copy_relative_path = f"{str(relative_path)[:-3]}_{i}.py"
        else:
            json_name = "removed.json"
            copy_path = f"{str(file_path)[:-3]}_removed.py"
            copy_relative_path = f"{str(relative_path)[:-3]}_removed.py"

        try:
            with open(copy_path, "w") as f:
                f.write(annotated_code)

            output_json = self.run_pyright(project_path, copy_relative_path)

            with open(output_path / json_name, "w") as f:
                f.write(output_json)
        finally:
            if os.path.exists(copy_path):
                os.remove(copy_path)

    def run(self):
        print(f"Running preprocess for {self.tool_name}...")

        data = self.load_data()

        for test_info in data:
            project_path = self.benchmark_path / test_info["repo_name"]

            file_path = project_path / test_info["file_path"]

            if not file_path.exists():
                test_info["file_path"] = "src/" + test_info["file_path"]
                file_path = project_path / test_info["file_path"]

            if not file_path.exists():
                continue

            folder_name = str(test_info["repo_name"] + "_" + test_info["file_path"])[:-3].replace("/", "_")

            output_path = Path("data") / self.tool_name / folder_name / test_info["target"].replace('.', '_')

            target_name = test_info["target"]
            params = test_info["params"]
            top_n_predictions = test_info["predictions"]
            expects = test_info["expects"]

            print(f"Processing {file_path} for target {target_name}...")

            with open(file_path, "r") as f:
                code = f.read()

            if self.is_single:
                param = params[0]
                output_path = output_path / param
            else:
                top_n_predictions = self.make_none_of_empty_expects(top_n_predictions, expects)

            if not output_path.exists():
                output_path.mkdir(parents=True)

            correct_idx = []
            incorrect_idx = []

            for i, predictions in enumerate(top_n_predictions):
                is_correct = True
                for pred, expect in zip(predictions, expects):
                    if expect == None:
                        continue
                    
                    try:
                        updated_pred = str(parse_type_str(pred).normalized())
                        updated_expect = str(parse_type_str(expect).normalized())
                    except Exception:
                        is_correct = False
                        break

                    if updated_pred == updated_expect:
                        continue
                    else:
                        is_correct = False
                        break
                if is_correct:
                    correct_idx.append(i)
                else:
                    incorrect_idx.append(i)

            info = {
                'correct': correct_idx,
                'incorrect': incorrect_idx,
            }

            with open(output_path / "info.json", "w") as f:
                json.dump(info, f, indent=4)

            file_info = {
                'file_path' : str(file_path),
                'sig_name' : target_name,
                'var_name' : params,
            }

            with open(output_path / "file_info.json", "w") as f:
                json.dump(file_info, f, indent=4)

            top_n_predictions = self.run_pre_analysis(code, top_n_predictions)
            self.make_annotator(code, target_name, params)

            datas = []
            for i, predictions in enumerate(top_n_predictions):
                datas.append((i, predictions, project_path, file_path, test_info["file_path"], params, output_path))

            # for removed ones
            datas.append((None, predictions, project_path, file_path, test_info["file_path"], params, output_path))
    
            try:
                backup_path = str(file_path) + ".backup"
                shutil.copy(file_path, backup_path)
                with Pool() as pool:
                    pool.map(self.process_single, datas)
            finally:
                if os.path.exists(backup_path):
                    shutil.move(backup_path, file_path)


class TigerPreAnalysis(PreAnalysis):
    tool_name = "Tiger"
    benchmark_path = Path("ManyTypes4Py/repos")
    is_single = True

    def make_annotator(self, code, target_name, params):
        self.code = code
        self.target_name = target_name
        self.params = params 
        
    def annotate(self, predictions, is_removed=False):
        root = ast.parse(self.code)
        locator = FunctionLocator()

        target_name_split = self.target_name.split(".")

        if len(target_name_split) == 2:
            self.class_name = target_name_split[0]
            self.method_name = target_name_split[1]
        elif len(target_name_split) == 1:
            self.class_name = "global"
            self.method_name = target_name_split[0]
        else:
            raise ValueError(f"Invalid target name: {self.target_name}")

        if self.params[0] == "__RET__":
            self.var_name = self.method_name
            self.kind = "return"
        else:
            self.var_name = self.params[0]
            self.kind = "arg"

        if is_removed:
            node = locator.run(root, f"{self.method_name}@{self.class_name}", self.var_name, self.kind)
            annotate_function(node, None, self.var_name, is_param=(self.kind=="arg"), is_removed=True)
        else:
            pred = predictions[0]
            if pred is None:
                return None

            try:
                parsed_type = str(parse_type_str(pred))
            except Exception:
                return None

            node = locator.run(root, f"{self.method_name}@{self.class_name}", self.var_name, self.kind)
            annotate_function(node, parsed_type, self.var_name, is_param=(self.kind=="arg"))

        annotated_code = ast.unparse(root)
        return annotated_code

    def run_pre_analysis(self, code, top_n_predictions):
        is_imported_optional = self.check_optional(code)
        if is_imported_optional:
            top_n_predictions = self.change_optional_type(top_n_predictions)

        return top_n_predictions

class TypeGenPreAnalysis(PreAnalysis):
    tool_name = "TypeGen"
    benchmark_path = Path("ManyTypes4Py/repos")
    is_single = True

    def make_annotator(self, code, target_name, params):
        self.code = code
        self.target_name = target_name
        self.params = params 
        
    def annotate(self, predictions, is_removed=False):
        root = ast.parse(self.code)
        locator = FunctionLocator()

        target_name_split = self.target_name.split(".")

        if len(target_name_split) == 2:
            self.class_name = target_name_split[0]
            self.method_name = target_name_split[1]
        elif len(target_name_split) == 1:
            self.class_name = "global"
            self.method_name = target_name_split[0]
        else:
            raise ValueError(f"Invalid target name: {self.target_name}")

        if self.params[0] == "__RET__":
            self.var_name = self.method_name
            self.kind = "return"
        else:
            self.var_name = self.params[0]
            self.kind = "arg"

        if is_removed:
            node = locator.run(root, f"{self.method_name}@{self.class_name}", self.var_name, self.kind)
            annotate_function(node, None, self.var_name, is_param=(self.kind=="arg"), is_removed=True)
        else:
            pred = predictions[0]
            if pred is None:
                return None

            try:
                parsed_type = str(parse_type_str(pred))
            except Exception:
                return None

            node = locator.run(root, f"{self.method_name}@{self.class_name}", self.var_name, self.kind)
            annotate_function(node, parsed_type, self.var_name, is_param=(self.kind=="arg"))

        annotated_code = ast.unparse(root)
        return annotated_code

    def run_pre_analysis(self, code, top_n_predictions):
        is_imported_optional = self.check_optional(code)
        if is_imported_optional:
            top_n_predictions = self.change_optional_type(top_n_predictions)

        return top_n_predictions

class ExamplePreAnalysis(PreAnalysis):
    tool_name = "Example"
    benchmark_path = Path(".")
    is_single = True

    def make_annotator(self, code, target_name, params):
        self.code = code
        self.target_name = target_name
        self.params = params 
        
    def annotate(self, predictions, is_removed=False):
        root = ast.parse(self.code)
        locator = FunctionLocator()

        target_name_split = self.target_name.split(".")

        if len(target_name_split) == 2:
            self.class_name = target_name_split[0]
            self.method_name = target_name_split[1]
        elif len(target_name_split) == 1:
            self.class_name = "global"
            self.method_name = target_name_split[0]
        else:
            raise ValueError(f"Invalid target name: {self.target_name}")

        if self.params[0] == "__RET__":
            self.var_name = self.method_name
            self.kind = "return"
        else:
            self.var_name = self.params[0]
            self.kind = "arg"

        if is_removed:
            node = locator.run(root, f"{self.method_name}@{self.class_name}", self.var_name, self.kind)
            annotate_function(node, None, self.var_name, is_param=(self.kind=="arg"), is_removed=True)
        else:
            pred = predictions[0]
            if pred is None:
                return None

            try:
                parsed_type = str(parse_type_str(pred))
            except Exception:
                return None

            node = locator.run(root, f"{self.method_name}@{self.class_name}", self.var_name, self.kind)
            annotate_function(node, parsed_type, self.var_name, is_param=(self.kind=="arg"))

        annotated_code = ast.unparse(root)
        return annotated_code

    def run_pre_analysis(self, code, top_n_predictions):
        is_imported_optional = self.check_optional(code)
        if is_imported_optional:
            top_n_predictions = self.change_optional_type(top_n_predictions)

        return top_n_predictions
    
class TypeT5PreAnalysis(PreAnalysis):
    tool_name = "TypeT5"
    benchmark_path = Path("BetterTypes4Py/repos/test")
    is_single = False

    def make_none_of_empty_expects(self, top_n_predictions, expects):
        none_idx = []
        for i, expect_type in enumerate(expects[:-1]):
            if expect_type == None:
                none_idx.append(i)

        new_top_n_predictions = []
        for predictions in top_n_predictions:
            new_predictions = []

            for i, pred in enumerate(predictions):
                if i in none_idx:
                    new_predictions.append(None)
                else:
                    new_predictions.append(pred)        
            
            new_top_n_predictions.append(new_predictions)
            
        return new_top_n_predictions

    def make_annotator(self, code, target_name, params):
        self.annotator = CSTAnnotator(code, target_name, params)

    def annotate(self, predictions, is_removed=False):
        return self.annotator.annotate(predictions, is_removed=is_removed)
    
    def run_pre_analysis(self, code, top_n_predictions):
        is_imported_optional = self.check_optional(code)
        if is_imported_optional:
            top_n_predictions = self.change_optional_type(top_n_predictions)

        return top_n_predictions

def run(tool_name):
    if tool_name == "tiger":
        pre_analysis = TigerPreAnalysis()
    elif tool_name == "typet5":
        pre_analysis = TypeT5PreAnalysis()
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")

    pre_analysis.run()


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--tool", "-t", type=str, required=True, choices=['typet5', 'tiger', 'typegen', 'example'], help="tool name <typet5|tiger|typegen|example>")
    args = argument_parser.parse_args()

    if args.tool == "typet5":
        TypeT5PreAnalysis().run()
    elif args.tool == "typegen":
        TypeGenPreAnalysis().run()
    elif args.tool == "tiger":
        TigerPreAnalysis().run()
    elif args.tool == "example":
        ExamplePreAnalysis().run()