from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import requests
from bs4 import BeautifulSoup
import random
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def obtener_imagenes_galeria():
    URL = "https://www.disycom.com.mx"
    CARPETA = "/wp-content/uploads/"
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [
        img['src'] for img in soup.find_all('img')
        if CARPETA in img.get('src', '')
    ]

@app.post("/generar-imagen-ia")
async def generar_imagen_ia(data: PromptRequest):
    prompt = data.prompt
    imagenes = obtener_imagenes_galeria()
    if not imagenes:
        return {"error": "No se encontraron imágenes"}

    imagen_url = random.choice(imagenes)
    response = requests.get(imagen_url)
    imagen = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

    resultado = pipe(prompt=prompt, image=imagen, strength=0.75, guidance_scale=7.5)
    generada = resultado.images[0]
    generada.save("static/imagen_generada.png")

    return {
        "original": imagen_url,
        "generada": "/static/imagen_generada.png"
    }
