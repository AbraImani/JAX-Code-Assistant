import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM

def load_model(model_name="google/t5-small"): # exemple sur le plus facile
    """Charge le modèle et le tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_response(prompt: str, model, tokenizer):
    """Génère une réponse basée sur une requête donnée."""
    inputs = tokenizer(prompt, return_tensors="jax", padding=True, truncation=True)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or "Comment utiliser jax.grad ?"
    tokenizer, model = load_model()
    response = generate_response(prompt, model, tokenizer)
    print("Réponse générée :", response)
