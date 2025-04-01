import numpy as np

def evaluate_model(predictions, targets):
    """
    Évalue le modèle en utilisant l'erreur quadratique moyenne (MSE).
    """
    mse = np.mean((predictions - targets) ** 2)
    return mse

def dummy_evaluation():
    # Prédictions et cibles fictives pour la démonstration
    predictions = np.array([1.0, 2.0, 3.0]) # à modifier 
    targets = np.array([1.5, 2.5, 3.5]) # sera modifié 
    mse = evaluate_model(predictions, targets)
    print("MSE :", mse)

if __name__ == "__main__":
    dummy_evaluation()
  
