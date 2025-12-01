from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def get_iris_data(classes=[0, 1]):
    data = load_iris()
    df = pd.DataFrame(data.data)
    df['class'] = data.target
    df = df[df['class'].isin(classes)]
    df.index.name = 'datetime'

    df_train, df_test = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["class"],
            shuffle = True
            )

    df_train.to_csv('../test_data_iris_train.txt', header=True, index=True, sep='\t')
    df_test.to_csv('../test_data_iris_test.txt', header=True, index=True, sep='\t')

get_iris_data([0, 1])
#get_iris_data([0, 1, 2])
