import pandas as pd
from sklearn.feature_selection import SelectKBest


def select_k_best(score, X, Y):
    selector = SelectKBest(score, k=20)
    selector.fit_transform(X, Y)
    names = X.columns.values[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    names_scores = list(zip(names, scores))
    df_reduced = pd.DataFrame(data=names_scores, columns=['feature_names', 'score'])

    df_reduced = df_reduced.sort_values(['score', 'feature_names'], ascending=[False, True])
    print(df_reduced)
    return df_reduced.feature_names
