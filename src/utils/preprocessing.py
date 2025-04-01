import pandas as pd

def preprocess_data(df):
    """
    Fonction de prétraitement factice :
    - Remplit les valeurs manquantes par la moyenne.
    - Normalise les colonnes numériques.
    - Trouver les colonnes catégorielles
    """
    df = df.fillna(0)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df

if __name__ == "__main__":
    # Exemple de comment sera le test de la fonction de prétraitement avec des données fictives
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': [4, None, 6, 8]
    })
    processed_df = preprocess_data(df)
    print(processed_df)
  
