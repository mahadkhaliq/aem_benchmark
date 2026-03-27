import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = os.path.join(os.path.dirname(__file__), 'models', 'MLP', 'adm_mlp')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')

os.makedirs(OUT_DIR, exist_ok=True)

ea = EventAccumulator(LOG_DIR)
ea.Reload()

for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    df = pd.DataFrame({'step': [e.step for e in events], 'value': [e.value for e in events]})
    fname = tag.replace('/', '_') + '.csv'
    df.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"Saved {fname} ({len(df)} rows)")
