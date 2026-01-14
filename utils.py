import json
from pathlib import Path
import libcst as cst
import ast
import re, os, csv
from hityper.typeobject import TypeObject
import shutil
import subprocess
from typet5.type_check import parse_type_str

def remove_prefix_filename(filename):
    filename = filename.split("/repos/")[-1]

    return filename

def remove_suffix_filename(filename, is_removed=False):
    if is_removed:
        if filename[-11:-3] == '_removed':
            filename = filename[:-11] + filename[-3:]  # Remove the last '_removed' if exists
    else:
        if filename[-5] == '_':
            filename = filename[:-5] + filename[-3:]
    
    return filename

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

def is_kwonlyargs(node, var_name, is_param):
    if is_param:
        for arg in node.args.kwonlyargs:
            if arg.arg == var_name:
                return True
    return False

def find_function_node(root, loc, name, scope):
    locator = FunctionLocator()
    return locator.run(root, loc, name, scope)

def annotate_function(func_node: ast.FunctionDef, typ: str, var_name, is_param=True, is_removed=False) -> ast.FunctionDef:
    # 파라미터에 타입 추가
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
        # 리턴 타입 추가
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
        # 리턴 타입 추가
        else:
            func_node.returns = ast.Name(id=typ, ctx=ast.Load())
    
    return func_node

def output_folder_name(repo_name, file_path):
    folder_name = str(repo_name + "_" + file_path)[:-3].replace("/", "_")
    return folder_name