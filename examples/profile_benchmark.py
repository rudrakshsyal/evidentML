import json

import pandas as pd

from evidentml.data_engine.profiler import profile_dataset

df = pd.read_csv("benchmarks/datasets/binary_classification_messy_v1.csv")

profile = profile_dataset(df)

print(json.dumps(profile, indent=2))
