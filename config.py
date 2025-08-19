from pathlib import Path

# Home Path
home = Path.home()
ours = home / "TypeCare"

# Prediction Path
PRED_PATH = ours / "prediction"

# Data Path
DATA_PATH = ours / "data"

# Output Path
OUTPUT_PATH = ours / "output"

# Model Path
MODEL_PATH = PRED_PATH / "typesim_model"


TIGER_RESULT_PATH = PRED_PATH / "Tiger" / "transformed_result.json"
TIGER_OUTPUT_PATH = OUTPUT_PATH / "tiger_result.pkl"


# Configuration for TypeGen
TYPEGEN_PATH= home / "TypeGen"
TYPEGEN_PREDICTION_PATH = TYPEGEN_PATH / "data" / "predictions" / "typegen.json"
TYPEGEN_TESTSET_PATH = PRED_PATH / "TypeGen" / "transformed_result.json"
TYPEGEN_OUTPUT_PATH = OUTPUT_PATH / "typegen_result.pkl"

# Configuration for TypeT5
TYPET5_PATH = home / "TypeT5"
TYPET5_PREDICTION_PATH = TYPET5_PATH / 'evaluations/ManyTypes4Py' / 'double-traversal-EvalResultAllTest_0329.pkl'
TYPET5_TRANSFORM_PATH = PRED_PATH / "TypeT5" / "typet5_function_transform.pkl"
TYPET5_OUTPUT_PATH = OUTPUT_PATH / "typet5_result.pkl"