# Extract File Path and Information from the given pickle file
from typet5.static_analysis import FunctionSignature
from typet5.type_check import parse_type_str, parse_type_expr, normalize_type

import pickle
from logger import logger
from config import RESULT_PATH, TYPET5_PREDICTION_PATH, TYPET5_TRANSFORM_PATH
from pathlib import Path
import os
import libcst as cst

def typet5_transform():
    transform_data = []

    logger.info("Loading TypeT5 prediction results...")

    with open(TYPET5_PREDICTION_PATH, 'rb') as f:
        evalr = pickle.load(f)
    
    prj_roots = evalr.project_roots

    logger.info("TypeT5 prediction results loaded: %i projects", len(prj_roots))

    label_maps = evalr.label_maps
    predictions = evalr.predictions
    result = evalr.return_predictions()

    for i, (proj, sig_map) in enumerate(result.items()):
        logger.info(f"({i+1}/{len(result)}) Processing {proj.name}")
        proj_name = proj.name

        for path, sig in sig_map.items():
            pid = evalr.find_projects(proj_name)[0]
            top_pred = evalr.predictions[pid].elem2preds[path]
            all_predcitions = evalr.predictions[pid].elem2all_preds[path]
            expected = evalr.label_maps[pid][path]
            
            if isinstance(sig, FunctionSignature):
                sig_name = str(path).split('/')[1]
                src_name = str(path).split('/')[0]
                src_path = src_name.replace('.', '/') + '.py'
                file_path_splited = str(proj).split("/")
                file_path = Path(str(proj).replace("/home/wonseok/", "/home/wonseokoh/"))
                if not os.path.exists(file_path / src_path):
                    file_path = file_path / 'src' / src_path
                else:
                    file_path = file_path / src_path

                repo_name = file_path_splited[-1]

                assert file_path_splited[-2] == "test"

                expected_params = expected.params
                expected_returns = expected.returns
                predicted_params = sig.params
                predicted_returns = sig.returns
                predicted_returns = str(parse_type_str(cst.Module([]).code_for_node(predicted_returns.annotation)).normalized())


                # print(expected_params)
                # input()

                # check expected_params is all None
                is_all_none = True
                for param in expected_params.values():
                    if param != None:
                        is_all_none = False
                        break

                if (is_all_none or (not expected_params)) and expected_returns == None:
                    continue

                param_infos = list(expected_params.items())
                predicted_params = list(predicted_params.keys())
                target_param_indexes = []

                for (param_key, typ) in param_infos:
                    if typ is None:
                        continue
                    if param_key in predicted_params:
                        target_param_indexes.append(predicted_params.index(param_key))

                top_pred = [str(pred.normalized()) for pred in top_pred]
                ret_idx = top_pred.index(predicted_returns)

                param_list = list(expected.params.keys())
                params_name = [param_list[i] for i in target_param_indexes]
                params_name.append("__RET__")
                
                expected_types = []

                for param, expected_type in expected_params.items():
                    if expected_type == None:
                        continue

                    code = cst.Module([]).code_for_node(expected_type.annotation)
                    expected_type = str(parse_type_str(code))
                    expected_types.append(expected_type)

                if expected_returns is not None:
                    expected_types.append(str(parse_type_expr(expected_returns.annotation)))
                else:
                    expected_types.append(None)

                predictions = []

                for preds in all_predcitions:
                    preds = [str(pred) for pred in preds]
                    if expected_returns is not None:
                        returns = preds[ret_idx]
                    else:
                        returns = None

                    preds = [preds[i] for i in target_param_indexes]
                    preds.append(returns)   
                    predictions.append(preds)

                    
                result_path = Path(str(proj_name)) / src_name.replace('.', '_') / sig_name.replace('.', '_')

                assert len(params_name) == len(predictions[0]) and len(params_name) == len(expected_types), \
                f"Length mismatch: {expected.params} {params_name}, {predictions[0]}, {expected_types}"

                transform_data.append({
                    "repo_name": str(proj_name),
                    "src_path": src_path,
                    "file_path": str(file_path),
                    "result_path": str(result_path),
                    "target": sig_name,
                    "params": params_name,
                    "predictions": predictions,
                    "expects": expected_types
                })

    with open(TYPET5_TRANSFORM_PATH, "wb") as f:
        pickle.dump(transform_data, f)

if __name__ == "__main__":
    typet5_transform()