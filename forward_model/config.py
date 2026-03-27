import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

LINEAR        = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
BATCH_SIZE    = 1024
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
LR_DECAY_RATE = 0.2
EVAL_STEP     = 10
STOP_THRESHOLD = 1e-7
NORMALIZE_INPUT = True
