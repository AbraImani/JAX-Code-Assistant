import streamlit as st
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM

def load_model(model_name="google/t5-small"):
    """Charge le modèle et le tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name) #replace par la modula 
    return tokenizer, model

st.title("JAX Code Assistant")
user_input = st.text_area("Entrez une question ou un code à compléter")
if st.button("Générer la réponse"):
    tokenizer, model = load_model()
    inputs = tokenizer(user_input, return_tensors="jax", padding=True, truncation=True)
    output = model.generate(**inputs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("### Réponse :")
    st.write(response)
