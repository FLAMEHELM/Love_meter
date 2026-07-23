import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ------------------------------------------------------------
# 1. Функция для создания/пересоздания модели
# ------------------------------------------------------------
def create_model():
    print("🔄 Создаю новую модель на синтетических данных...")
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
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Модель и скейлер созданы и сохранены.")
    return model, scaler

# ------------------------------------------------------------
# 2. Загрузка или создание модели (с обработкой ошибок)
# ------------------------------------------------------------
model = None
scaler = None

try:
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Модель и скейлер загружены из файлов.")
    else:
        model, scaler = create_model()
except Exception as e:
    print(f"⚠️ Ошибка загрузки модели: {e}")
    print("🔄 Пересоздаю модель...")
    # Удаляем повреждённые файлы, если они есть
    if os.path.exists('model.pkl'):
        os.remove('model.pkl')
    if os.path.exists('scaler.pkl'):
        os.remove('scaler.pkl')
    model, scaler = create_model()

# ------------------------------------------------------------
# 3. FastAPI приложение
# ------------------------------------------------------------
app = FastAPI(title="Счетчик Любви ❤️", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Здесь вставь свой HTML (я его опускаю для краткости, ты его уже имеешь)
HTML_PATH = "templates/index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_PATH)

@app.get("/predict")
async def predict(messages: int, response_time: int, emojis: int, first_msg: int, positive_ratio: float, is_male: int):
    features = np.array([[messages, response_time, emojis, first_msg, positive_ratio, is_male]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    pred = round(pred, 1)
    
    # комментарии...
    if pred >= 130:
        comment = "💖 Божественная любовь!"
    elif pred >= 100:
        comment = "❤️ Очень сильная любовь!"
    elif pred >= 70:
        comment = "💕 Хорошая любовь!"
    elif pred >= 40:
        comment = "🧡 Теплые чувства!"
    else:
        comment = "💔 Нужно поработать над отношениями."
    who = "Она любит тебя" if is_male == 1 else "Ты любишь её"
    return {"love_percent": pred, "message": f"{who} на {pred}%! {comment}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
