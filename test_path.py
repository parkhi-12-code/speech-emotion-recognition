import os

DATA_PATH = "data/Ravdess/"   # or "data/" if you didn't create RAVDESS folder

for root, dirs, files in os.walk(DATA_PATH):
    for file in files[:5]:
        print(os.path.join(root, file))