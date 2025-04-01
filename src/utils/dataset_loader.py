import numpy as np

def load_dataset():
    """
    Charge un dataset factice.
    Renvoie :
        X : Features sous forme de tableau NumPy.
        y : Cibles sous forme de tableau NumPy.
    """
    # Comme pas encore eu des datasets fictives utilisons d'un dataset aléatoire pour la démonstration (100 échantillons, 5 features)
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)
    return X, y

if __name__ == "__main__":
    X, y = load_dataset()
    print("Forme de X :", X.shape)
    print("Forme de y :", y.shape)
  
