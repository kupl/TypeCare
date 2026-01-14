from pathlib import Path

# Home Path
HOME_PATH = Path.home()
ours = HOME_PATH / "TypeCare"

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
TYPEGEN_PATH= PRED_PATH / "TypeGen"
TYPEGEN_PREDICTION_PATH = TYPEGEN_PATH / "data" / "predictions" / "typegen.json"
TYPEGEN_TESTSET_PATH = PRED_PATH / "TypeGen" / "transformed_result.json"
TYPEGEN_OUTPUT_PATH = OUTPUT_PATH / "typegen_result.pkl"

# Configuration for TypeT5
TYPET5_PATH = PRED_PATH / "TypeT5"
TYPET5_PREDICTION_PATH = PRED_PATH / "TypeT5" / 'double-traversal-EvalResultAllTest_0329.pkl'
TYPET5_TRANSFORM_PATH = PRED_PATH / "TypeT5" / "transformed_result.json"
TYPET5_OUTPUT_PATH = OUTPUT_PATH / "typet5_result.pkl"