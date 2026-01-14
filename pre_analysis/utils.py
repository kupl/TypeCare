import libcst as cst
import ast

class TypingOptionalCheckerLegacy(cst.CSTVisitor):
    """
    check whether 'Optional' is imported from 'typing' module
    """
    def __init__(self):
        super().__init__()
        self.found_optional_import: bool = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module_is_typing = (
            node.module is not None and
            isinstance(node.module, cst.Name) and
            node.module.value == "typing"
        )

        if module_is_typing:
            if isinstance(node.names, cst.ImportStar):
                self.found_optional_import = True
                return
            
            for import_alias in node.names:
                is_optional_imported = (
                    isinstance(import_alias.name, cst.Name) and
                    import_alias.name.value == "Optional"
                )

                if is_optional_imported:
                    self.found_optional_import = True
                    return
                

class CSTAnnotator():
    def __init__(self, code, target_name, params):
        self.code = code
        self.target_name = target_name
        self.params = params

        module = cst.parse_module(code)
        self.wrapper = cst.metadata.MetadataWrapper(module)

    def annotate(self, predictions, is_removed=False):
        if not predictions:
            return None

        prediction_of_params = {}
        prediction_of_return = None

        for param, pred in zip(self.params, predictions):
            if pred is None:
                continue
            try:
                pred = cst.Annotation(cst.parse_expression(pred))
            except Exception:
                return None

            if param == "__RET__":
                prediction_of_return = pred
            else:
                prediction_of_params[param] = pred


        if is_removed:
            remove_transformer = RemoveAnnotationTransformer(self.target_name, prediction_of_params, prediction_of_return)
            remove_module = self.wrapper.visit(remove_transformer)

            return remove_module.code
        else:
            override_transformer = OverrideAnnotationTransformer(self.target_name, prediction_of_params, prediction_of_return)
            override_module = self.wrapper.visit(override_transformer)

            return override_module.code

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