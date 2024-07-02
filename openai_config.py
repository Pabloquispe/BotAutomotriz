# openai_config.py
import openai
import os

# Configuración de la API de OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def interactuar_con_openai(pregunta):
    response = openai.Completion.create(
        engine="davinci",
        prompt=pregunta,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.9,
    )
    return response.choices[0].text.strip()
