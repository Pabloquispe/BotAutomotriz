import openai
import os

# Configuración de la API de OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def interactuar_con_openai(consulta):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": consulta}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError:
        return "❌ **Lo siento, hemos superado nuestro límite de solicitudes por ahora. Por favor, intenta de nuevo más tarde.**"
    except openai.error.OpenAIError as e:
        print(f"Error interacting with OpenAI: {e}")
        return "❌ **Ha ocurrido un error al interactuar con OpenAI. Por favor, intenta de nuevo más tarde.**"

