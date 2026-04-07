import pandas as pd
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(project_root, "data", "train.csv"))
test_df = pd.read_csv(os.path.join(project_root, "data", "test.csv"))

print("TRAIN DATA")
print(train_df.head())
print(train_df.info())

print("\nTEST DATA")
print(test_df.head())
print(test_df.info())