from fastapi import FastAPI
from pydantic import BaseModel
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM

app = FastAPI()
model_name = "google/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name) # Sera change par la modularité de l'appel du modèle entraîné.
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name)

class QueryRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_text(request: QueryRequest):
    """Génère une réponse à une requête en POST."""
    inputs = tokenizer(request.query, return_tensors="jax", padding=True, truncation=True)
    output = model.generate(**inputs)
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}
