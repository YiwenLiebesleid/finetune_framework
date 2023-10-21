"""
    pipeline for whisper tasks
    1.1 configuration (go to frame_config.py)
    1.2 data preparation (go to prepare_data.py)
    2. model training
    3. model evaluation

    you should do with 1.1 and 1.2
"""
from debug_util import debug_prt
from frame_config import pipeline_stage_setup
from prepare_data import udf_prep_data
from model_train import model_training
from evaluation import do_evaluation


def run_pipeline():
    debug_prt("Pipeline start", "[PIPELINE]")
    print()

    if pipeline_stage_setup["data_preparation"] is True:
        debug_prt("Start data preparation...", "[PIPELINE]")
        udf_prep_data()
    else:
        debug_prt("not doing data preparation", "[PIPELINE]")
    print()

    if pipeline_stage_setup["model_training"] is True:
        debug_prt("Start model training...", "[PIPELINE]")
        model_training()
    else:
        debug_prt("not doing model training", "[PIPELINE]")
    print()
    
    if pipeline_stage_setup["evaluation"] is True:
        debug_prt("Start evaluation...", "[PIPELINE]")
        do_evaluation()
    else:
        debug_prt("not doing evaluation", "[PIPELINE]")
    print()

    debug_prt("Pipeline finished", "[PIPELINE]")


if __name__ == "__main__":
    run_pipeline()