import os
import libcst as cst
import shutil
import subprocess
from logger import logger

class OverrideAnnotationTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)  # ParentNodeProvider 추가

    def __init__(self, target_name: str, annotations: dict, return_annotation: str):
        # 타겟 함수 이름, 인수 주석 및 반환 주석을 초기화
        if "." in target_name:
            splited = target_name.split(".")
            self.target_class, self.target_function = splited[:-1], splited[-1]
        else:
            self.target_class = None
            self.target_function = target_name
        self.annotations = annotations
        self.return_annotation = return_annotation

    def _is_in_target_class(self, node: cst.FunctionDef) -> bool:
        # 주어진 함수 노드가 target_class 안에 있는지 확인하는 내부 함수
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
        # 클래스 안의 메서드일 경우
        if self.target_class:
            # 현재 함수가 target_class 내부에 있는지 확인
            if not self._is_in_target_class(original_node):
                return updated_node

        # 함수 이름이 타겟 함수와 일치할 경우에만 주석을 덮어씌움
        if original_node.name.value == self.target_function:
            # 각 파라미터에 대해 새로운 주석 설정
            is_updated = False
            new_params = []
            for param in updated_node.params.params:
                param_name = param.name.value
                if param_name in self.annotations:
                    # 새 주석 적용
                    new_annotation = cst.Annotation(cst.parse_expression(self.annotations[param_name]))
                    new_param = param.with_changes(annotation=new_annotation)
                    is_updated = True
                else:
                    # 해당하는 주석이 없으면 그대로 유지
                    new_param = param.with_changes(annotation=None)
                new_params.append(new_param)

            param_keys = self.annotations.keys()
            if (not is_updated) and param_keys != ["__RET__"]:
                return updated_node

            # 반환 타입 주석 덮어쓰기
            if self.return_annotation:
                new_return_annotation = cst.Annotation(cst.parse_expression(self.return_annotation))
            else:
                new_return_annotation = None

            updated_node = updated_node.with_changes(
                params=updated_node.params.with_changes(params=new_params),
                returns=new_return_annotation
            )

        return updated_node

def run_static_analysis_typet5(args):
    k, proj_path, file_path, result_path, func_name, pred_params, pred_return, output_file = args

    # if os.path.exists(result_path / output_file):
    #     #logger.debug(f'{output_file} already exists')
    #     # pass
    #     return
    
    with open(proj_path / file_path, 'r') as f:
        code = f.read()

    module = cst.parse_module(code)
    wrapper = cst.metadata.MetadataWrapper(module)

    transformer = OverrideAnnotationTransformer(func_name, pred_params, pred_return)
    cand_module = wrapper.visit(transformer)

    file_path = str(file_path)[:-3] + f'_{k}.py'
    
    # replace function signature
    with open(proj_path / file_path, 'w') as f:
        f.write(cand_module.code)

    # analysis original code
    logger.debug(f'Run static analysis for {file_path}')
    p = subprocess.Popen(
        ['pyright', '--outputjson', str(file_path)], 
        stdout=subprocess.PIPE,
        cwd=str(proj_path)
    )
    output, err = p.communicate()
    output_json = output.decode('utf-8')

    # save original analysis result
    with open(result_path / output_file, 'w') as f:
        f.write(output_json)