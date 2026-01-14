from typing import List, Optional, Dict, Any
import ast

AST_CACHE = {} # File path -> Function name -> AST

def find_function_body(source_code: str, target_funcname: str, target_params: List[str]) -> Optional[List[ast.stmt]]:
    tree = ast.parse(source_code)

    if target_params[-1] == "__RET__":
        target_params = target_params[:-1]

    def match_function(node: ast.FunctionDef, expected_params: List[str]) -> bool:
        actual_params = [arg.arg for arg in node.args.args]
        actual_params += [arg.arg for arg in node.args.kwonlyargs]
        actual_params += [node.args.vararg.arg] if node.args.vararg else []
        actual_params += [node.args.kwarg.arg] if node.args.kwarg else []
        # Check if the function name matches

        return (set(expected_params) - set(actual_params)) == set()

    def visit(node, path_prefix=""):
        results = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef)):
                full_path = f"{path_prefix}{child.name}"
                if full_path == target_funcname and match_function(child, target_params):
                    results.append(child)
            elif isinstance(child, ast.ClassDef):
                class_name = f"{path_prefix}{child.name}."
                results.extend(visit(child, path_prefix=class_name))
        return results

    class FunctionFinder(ast.NodeVisitor):
        def __init__(self):
            self.matches = []
            self.path_stack = []

        def visit_ClassDef(self, node):
            self.path_stack.append(node.name)
            self.generic_visit(node)
            self.path_stack.pop()

        def visit_FunctionDef(self, node):
            full_path = '.'.join(self.path_stack + [node.name])
            if full_path == target_funcname and match_function(node, target_params):
                self.matches.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)  # Treat async funcs the same way

    finder = FunctionFinder()
    finder.visit(tree)


    # matches = visit(tree)
    if finder.matches:
        # assert len(matches) == 1, f"Multiple matches found for the function\n{source_code}\n{target_funcname}\n{target_params}"
        return finder.matches[-1]  # return last match
    else:
        pass
        # print(source_code)
        # print(target_funcname)
        # print(target_params)
        # raise ValueError(f"Function {target_funcname} with parameters {target_params} not found in the provided source code.")
    return None

def get_ast_cache(proj_path, file_path: str, func_name: str, params: List[str]) -> ast.FunctionDef:
    if file_path not in AST_CACHE:
        AST_CACHE[file_path] = {}

    if func_name not in AST_CACHE[file_path]:
        with open(proj_path / file_path, 'r') as file:
            file_content = file.read()
            try:
                matched_function = find_function_body(file_content, func_name, params)
                AST_CACHE[file_path][func_name] = matched_function
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                AST_CACHE[file_path][func_name] = None

    return AST_CACHE[file_path][func_name]