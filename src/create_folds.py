import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../data/Cleaned data/clean_data.csv")

    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True)
    y = df.status 

    for f, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_idx, "kfold"] = f

    df.to_csv('../data/fold_data/df_folds.csv', index=False)