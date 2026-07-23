import numpy as np
import pandas as pd 
import pickle 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os

from fastapi.middleware.cors import CORSMiddleware


model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    print("🔄 Обучение модели на синтетических данных...")
    np.random.seed(42)
    n_samples = 2000
    messages = np.random.randint(0, 51, n_samples)
    response_time = np.random.randint(10, 300, n_samples)
    emojis = np.random.randint(0, 21, n_samples)
    first_msg = np.random.randint(0, 2, n_samples)
    positive_ratio = np.round(np.random.uniform(0.2, 1.0, n_samples), 2)
    is_male = np.random.randint(0, 2, n_samples)
    
    X = pd.DataFrame({
        'messages': messages,
        'response_time': response_time,
        'emojis': emojis,
        'first_msg': first_msg,
        'positive_ratio': positive_ratio,
        'is_male': is_male
    })
    
    love = (50 + 0.8*messages - 0.2*response_time + 1.5*emojis + 15*first_msg + 35*positive_ratio + 5*is_male + np.random.normal(0,5,n_samples))
    y = np.clip(love, 0, 150)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Модель и скейлер созданы и сохранены.")
else:
    print("📂 Загрузка существующей модели и скейлера...")


with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


app = FastAPI(title='Счетчик Любви ❤️', version='1.0')

HTML_FILE = os.path.join("templates", "index.html")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/predict")
async def predict(
      messages: int, 
      response_time: int, 
      emojis: int, 
      first_msg: int, 
      positive_ratio: float,
      is_male: int
): 
    features = np.array([[messages, response_time, emojis, first_msg, positive_ratio, is_male]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    prediction = round(prediction, 1)

    if prediction >= 130:
        comment = "💖 Божественная любовь! Вы созданы друг для друга!"
    elif prediction >= 100:
        comment = "❤️ Очень сильная любовь! Она/Он точно в вас души не чает."
    elif prediction >= 70:
        comment = "💕 Хорошая любовь, есть над чем работать, но все отлично!"
    elif prediction >= 40:
        comment = "🧡 Теплые чувства есть, но пора добавить романтики!"
    else:
        comment = "💔 Нужно срочно поговорить и всё исправить!"

    if is_male == 1:
         who = 'Она любит тебя'
    else: 
         who = 'Ты любишь её'

    return {
         'love_percent': prediction,
         'message': f"{who} на {prediction}%! {comment}"
    }

if __name__ == "__main__": 
     uvicorn.run(app, host="127.0.0.1", port=8000)


