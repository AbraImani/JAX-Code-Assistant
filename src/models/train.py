import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from src.utils.dataset_loader import load_dataset

class DummyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Une couche dense simple pour la démonstration
        x = nn.Dense(features=10)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

def train_model():
    # Chargement d'un dataset factice, améliorer à suivre en cas d'un dossier des dataset 
    X, y = load_dataset()
    
    # Initialisation du modèle
    model = DummyModel()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, X.shape[1])))
    
    # Définition de l'optimiseur
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    # Fonction de calcul  d'erreur ou perte (MSE)
    def loss_fn(params, x, y):
        preds = model.apply(params, x)
        return jnp.mean((preds - y) ** 2)
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Boucle d'entraînement simple sur 5 epochs comme essaie
    for epoch in range(5):
        params, opt_state, loss = train_step(params, opt_state, X, y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

if __name__ == "__main__":
    train_model()
  
