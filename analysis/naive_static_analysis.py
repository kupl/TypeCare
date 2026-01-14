import ast
from collections import defaultdict, Counter
import glob
from typet5.type_check import parse_type_str, normalize_type
from logger import logger

class CtxInfo:
    def __init__(self, is_return):
        self.parent_type = None
        self.info = None
        self.is_return = is_return

    def __repr__(self):
        return f"<parent_type: {self.parent_type}, info: {self.info}>"

    def __str__(self):
        return f"<parent_type: {self.parent_type}, info: {self.info}>"

    def __eq__(self, other):
        return self.parent_type == other.parent_type and self.info == other.info

    def __hash__(self):
        return hash((self.parent_type, self.info))

    def update_ctx(self, node: ast.AST):
        self.parent_type = type(node).__name__

        if isinstance(node, ast.Call):
            self.info = ast.unparse(node.func)
        elif isinstance(node, ast.Attribute):
            if self.is_return:
                self.info = ast.unparse(node)
            else:
                if isinstance(node.value, ast.Name):
                    self.info = node.attr
                elif isinstance(node.value, ast.Attribute):
                    logger.debug(f"IMPOSSIBLE!! Attribute: {ast.unparse(node)}")

                # self.info = node.value.attr
        elif isinstance(node, (ast.Name, ast.Constant)):
            self.info = node.id if isinstance(node, ast.Name) else node.value
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if not self.is_return:
                self.info = str([ast.unparse(elt) for elt in node.elts])
        else:
            self.info = None

    @staticmethod
    def transform_ast_to_ctx(node: ast.AST, is_return=False):
        ctx_info = CtxInfo(is_return)
        ctx_info.update_ctx(node)
        return ctx_info

class CtxTypes:
    def __init__(self):
        self.param_ctx_types = defaultdict(dict)
        self.ret_ctx_types = defaultdict(dict)
        self.decorators = []

    def __str__(self):
        return f"param_ctx_types: {self.param_ctx_types}\n\nret_ctx_types: {self.ret_ctx_types}"

    def update_param_ctx(self, arg, filename, class_hierarchy, funcname, args, annotation, usages):
        if annotation is None:
            return

        param_ctx_type = self.param_ctx_types[arg]

        name_info = (filename, class_hierarchy, funcname, args)
        param_ctx_type_by_name_info = param_ctx_type.get(name_info, {})
        # for usage in usages:
        #     usage_ctx = CtxInfo.transform_ast_to_ctx(usage)
        #     types = param_ctx_type.get(usage_ctx, Counter())
        #     types[annotation] += 1
        #     param_ctx_type[usage_ctx] = types

        usages = frozenset([CtxInfo.transform_ast_to_ctx(usage) for usage in usages])
        types = param_ctx_type_by_name_info.get(usages, Counter())
        types[annotation] += 1
        param_ctx_type_by_name_info[usages] = types
        param_ctx_type[name_info] = param_ctx_type_by_name_info
        
        self.param_ctx_types[arg] = param_ctx_type

    def update_ret_ctx(self, func_name, filename, class_hierarchy, args, annotation, usages):
        if annotation is None:
            return

        ret_ctx_type = self.ret_ctx_types[func_name]

        name_info = (filename, class_hierarchy, func_name, args)
        ret_ctx_type_by_name_info = ret_ctx_type.get(name_info, {})

        usages = frozenset([CtxInfo.transform_ast_to_ctx(usage, is_return=True) for usage in usages])
        types = ret_ctx_type_by_name_info.get(usages, Counter())
        types[annotation] += 1

        ret_ctx_type_by_name_info[usages] = types
        ret_ctx_type[name_info] = ret_ctx_type_by_name_info

        self.ret_ctx_types[func_name] = ret_ctx_type

    def update_decorators(self, decorators):
        self.decorators.extend(decorators)

class ProjectUsageInfos:
    def __init__(self):
        self.project_usage_infos = defaultdict(dict)

    def update_usage_infos(self, proj_path):
        if proj_path in self.project_usage_infos:
            return

        usage_infos = {}

        for file in glob.glob(f"{proj_path}/**/*.py", recursive=True):
            try:
                with open(file, 'r') as f:
                    code = f.read()
                usage_info = analyze_code(code)
                usage_infos[file] = usage_info
            except Exception as e:
                continue

        self.project_usage_infos[proj_path] = usage_infos

    def extract_ctx_types(self, proj_path, filename, target_class, target_func):
        usage_infos = self.project_usage_infos[proj_path]
        ctx_types = CtxTypes()
        for file, usage_info in usage_infos.items():
            for class_hierarchy, func_infos in usage_info.items():
                for func_name, info in func_infos.items():
                    if str(proj_path / filename) == file and func_name == target_func and class_hierarchy == target_class:
                        continue
                    # if not (func_name == "add_nodes" or func_name == "test_ns_completion"):
                    #     continue

                    # print(func_name)
                    args = tuple(info["Args"].keys())
                    for arg, arg_info in info["Args"].items():
                        ctx_types.update_param_ctx(arg, file, class_hierarchy, func_name, args, arg_info["annotation"], arg_info["usage"])
                    ctx_types.update_ret_ctx(func_name, file, class_hierarchy, args, info["Ret"]["annotation"], info["Ret"]["usage"])
                    ctx_types.update_decorators(info["Decorators"])

        return ctx_types

    def extract_target_ctx_types(self, proj_path, filename, target_class, target_func):
        usage_infos = self.project_usage_infos[proj_path]

        ctx_types = CtxTypes()
        for file, usage_info in usage_infos.items():
            if file != str(proj_path / filename):
                continue
            for class_hierarchy, func_infos in usage_info.items():
                if class_hierarchy != target_class:
                    
                    continue

                for func_name, info in func_infos.items():
                    if func_name != target_func:
                        continue
                    args = tuple(info["Args"].keys())
                    for arg, arg_info in info["Args"].items():
                        ctx_types.update_param_ctx(arg, file, class_hierarchy, func_name, args, "__NO_ANNOTATION__", arg_info["usage"])
                    ctx_types.update_ret_ctx(func_name, file, class_hierarchy, args, "__NO_ANNOTATION__", info["Ret"]["usage"])
                    ctx_types.update_decorators(info["Decorators"])

        return ctx_types

class FunctionAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = defaultdict(dict)  # {class_hierarchy: {func_name: {args, ret}}}
        self.class_stack = []
        self.current_function = None
        self.arg_first_usage = {}  # {arg_name: first_usage_expr}

    def normalize(self, ty):
        return str(normalize_type(parse_type_str(ty)))
    
    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
    
    def visit_FunctionDef(self, node):
        class_hierarchy = ".".join(self.class_stack)
        self.current_function = node.name
        
        args_info = {}
        self.arg_first_usage = {}
        for arg in node.args.args:
            annotation = ast.unparse(arg.annotation) if arg.annotation else None
            if annotation:
                # annotation = self.normalize(annotation)
                annotation = annotation
            args_info[arg.arg] = {"annotation": annotation, "usage": []}

        if node.args.vararg:
            annotation = ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else None
            if annotation:
                # annotation = self.normalize(annotation)
                annotation = annotation
            args_info[node.args.vararg.arg] = {"annotation": annotation, "usage": []}

        if node.args.kwarg:
            annotation = ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else None
            if annotation:
                # annotation = self.normalize(annotation)
                annotation = annotation
            args_info[node.args.kwarg.arg] = {"annotation": annotation, "usage": []}

        for arg in node.args.kwonlyargs:
            annotation = ast.unparse(arg.annotation) if arg.annotation else None
            if annotation:
                # annotation = self.normalize(annotation)
                annotation = annotation
            args_info[arg.arg] = {"annotation": annotation, "usage": []}
        
        ret_annotation = ast.unparse(node.returns) if node.returns else None
        if ret_annotation:
            # ret_annotation = self.normalize(ret_annotation)
            ret_annotation = ret_annotation
        
        decorators = []
        filter_decorators = [
            "staticmethod",
            "classmethod",
            "property",
            "cached_property",
            "lru_cache",
            "dataclass",
            "overload",
            "abstractmethod",
            "pytest",
            "mock.patch",
            "parametrize"
        ]

        for decorator in node.decorator_list:
            deco_name = ast.unparse(decorator)
            
            is_add = True
            for filter_decorator in filter_decorators:
                if filter_decorator in deco_name:
                    is_add = False
                    break

            if is_add:
                decorators.append(deco_name)

        self.classes[class_hierarchy][self.current_function] = {
            "Decorators": decorators,
            "Args": args_info,
            "Ret": {"annotation": ret_annotation, "usage": []},
        }
        
        self.generic_visit(node)
        self.current_function = None
    
    def visit_Return(self, node):
        if self.current_function and node.value:
            class_hierarchy = ".".join(self.class_stack)
            # if isinstance(node.value, (ast.Call, ast.Attribute, ast.Constant)):
            if self.current_function not in self.classes[class_hierarchy]:
                self.classes[class_hierarchy][self.current_function] = {
                    "Decorators": [],
                    "Args": {}, 
                    "Ret": {"annotation": None, "usage": []}
                }
            self.classes[class_hierarchy][self.current_function]["Ret"]["usage"].append(node.value)
        self.generic_visit(node)
    
    def generic_visit(self, node):
        if self.current_function and isinstance(node, ast.expr):
            class_hierarchy = ".".join(self.class_stack)
            before_node = None
            for child in ast.iter_child_nodes(node):
                if isinstance(node, ast.IfExp) and child != node.test:
                    continue

                if isinstance(child, ast.Name):
                    if self.current_function not in self.classes[class_hierarchy]:
                        self.classes[class_hierarchy][self.current_function] = {
                            "Decorators": [],
                            "Args": {}, 
                            "Ret": {"annotation": None, "usage": []}
                        }


                    if child.id in self.classes[class_hierarchy][self.current_function]["Args"]:
                        # if child.id not in self.arg_first_usage:
                        #     self.arg_first_usage[child.id] = node
                        self.classes[class_hierarchy][self.current_function]["Args"][child.id]["usage"].append(node)

        super().generic_visit(node)

def analyze_code(code):
    try:
        tree = ast.parse(code)
        analyzer = FunctionAnalyzer()
        analyzer.visit(tree)
        return analyzer.classes
    except SyntaxError as e:
        # logger.error(f"Syntax error in code: {e}")
        return {}

if __name__ == "__main__":
    code = """
class OuterClass:
    class InnerClass:
        def this_is_function(x: int, y:str, b) -> int:
            b.x # this is not a usage for x
            d = x + 1
            z = foo(x, y)
            c = goo(x)

            if z:
                return None
            if c:
                return hoo()
            return x
    """
    
    result = analyze_code(code)
    from pprint import pprint
    pprint(result)
