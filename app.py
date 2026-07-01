from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import json
from datetime import datetime
import os
import traceback

# --- ЗАГРУЗКА МОДЕЛИ YOLOv8 ---
MODEL_PATH = "C:/Users/User/Desktop/mtuci_practica/runs/detect/hanger_detection/train_run-final-25/weights/best.pt"

model = None
HANGER_CLASS_NAME = "hanger"

try:
    print(f"Loading model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found. Fallback to 'yolov8n.pt'.")
        MODEL_PATH = 'yolov8n.pt'
        model = YOLO(MODEL_PATH)
        print("Fallback pre-trained model loaded.")
    else:
        model = YOLO(MODEL_PATH)
        print("Custom trained model loaded successfully.")

    print(f"Model class names: {model.names}")

    available_classes = list(model.names.values()) if isinstance(model.names, dict) else model.names

    if HANGER_CLASS_NAME not in available_classes:
        print(f"Critical error: Class '{HANGER_CLASS_NAME}' not found.")
        model = None

except Exception as e:
    print(f"Critical model loading error: {e}")
    traceback.print_exc()
    model = None

# --- НАСТРОЙКА ПРИЛОЖЕНИЯ FASTAPI ---
app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- ИНТЕГРИРОВАННЫЙ HTML-КОД ---
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Анализ вешалок в гардеробе</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: 'Segoe UI', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .upload-section { text-align: center; margin-bottom: 30px; }
        input[type="file"] { display: block; margin: 20px auto; padding: 10px; border: 1px solid #ccc; border-radius: 5px; width: 60%; }
        button { background-color: #3498db; color: white; padding: 12px 25px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #2980b9; }
        #resultImage { display: block; margin: 30px auto; max-width: 100%; border: 1px solid #eee; border-radius: 5px; }
        #stats { text-align: center; margin-top: 20px; font-size: 1.2em; font-weight: bold; color: #2c3e50; }
        #historyList { list-style: none; padding: 0; margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; }
        #historyList li { background-color: #f9f9f9; padding: 15px; margin-bottom: 15px; border-radius: 5px; border: 1px solid #e0e0e0; display: flex; justify-content: space-between; align-items: center; }
        #historyList li a { color: #3498db; text-decoration: none; font-weight: bold; }
        .error-message { color: #e74c3c; text-align: center; font-weight: bold; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Анализ вешалок в гардеробе</h1>
        <div class="upload-section">
            <input type="file" id="wardrobeImage" accept="image/*">
            <button onclick="analyzeWardrobe()">Проверить вешалки</button>
        </div>

        <img id="resultImage" src="" alt="Результат анализа" style="display: none;">
        <div id="stats"></div>
        <div id="errorMessage" class="error-message" style="display: none;"></div>

        <h2>История анализа</h2>
        <ul id="historyList"></ul>
    </div>

    <script>
        async function analyzeWardrobe() {
            const fileInput = document.getElementById('wardrobeImage');
            const file = fileInput.files; 

            const resultImage = document.getElementById('resultImage');
            const statsDiv = document.getElementById('stats');
            const errorDiv = document.getElementById('errorMessage');

            resultImage.style.display = 'none';
            statsDiv.innerText = '';
            errorDiv.style.display = 'none';

            if (!file) {
                errorDiv.style.display = 'block';
                errorDiv.innerText = 'Пожалуйста, выберите файл изображения.';
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/analyze_wardrobe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: "Не удалось получить данные об ошибке." }));
                    throw new Error(`Ошибка сервера: ${response.status} - ${errorData.error || errorData.message || "Неизвестная ошибка."}`);
                }

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                resultImage.src = data.result_image_url + '?' + Date.now();
                resultImage.style.display = 'block';
                statsDiv.innerText = `Найдено вешалок: ${data.count} (Всего объектов: ${data.total_objects_detected})`;

                loadHistory();

            } catch (error) {
                console.error("Ошибка при анализе:", error);
                errorDiv.style.display = 'block';
                errorDiv.innerText = `Произошла ошибка: ${error.message}`;
                statsDiv.innerText = '';
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch('/history');
                if (!response.ok) {
                    throw new Error(`Ошибка загрузки истории: ${response.status}`);
                }
                const data = await response.json();
                const historyList = document.getElementById('historyList');
                historyList.innerHTML = '';

                if (data.history && data.history.length > 0) {
                    data.history.forEach(item => {
                        const listItem = document.createElement('li');
                        const formattedTimestamp = new Date(item.timestamp).toLocaleString();
                        listItem.innerHTML = `
                            <span>
                                <strong>${formattedTimestamp}</strong><br>
                                Файл: ${item.input_filename}, Вешалок: ${item.found_hangers}
                            </span>
                            <a href="${item.result_image_url}" target="_blank">Посмотреть результат</a>
                        `;
                        historyList.appendChild(listItem);
                    });
                } else {
                    historyList.innerHTML = '<li>Пока нет данных об анализе.</li>';
                }
            } catch (error) {
                console.error("Ошибка при загрузке истории:", error);
                document.getElementById('historyList').innerHTML = `<li>Не удалось загрузить историю: ${error.message}</li>`;
            }
        }

        window.onload = loadHistory;
    </script>
</body>
</html>
"""


# --- МАРШРУТЫ FASTAPI ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if model is None:
        return HTMLResponse("<h1>Ошибка загрузки модели!</h1><p>Проверьте путь к best.pt</p>", status_code=500)
    return HTMLResponse(content=HTML_CONTENT)


@app.post("/analyze_wardrobe")
async def analyze_wardrobe(request: Request, image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение.")

        results = model.predict(img, conf=0.3, iou=0.5)

        if not results or len(results) == 0:
            raise HTTPException(status_code=500, detail="Модель вернула пустой результат.")


        r = results

        if not hasattr(r, 'plot') or not hasattr(r, 'boxes'):
            raise HTTPException(status_code=500, detail="Неверный формат ответа от модели.")

        hanger_boxes = []
        total_objects_detected = 0

        if r.boxes is not None:
            total_objects_detected = len(r.boxes)
            for box in r.boxes:
                class_id = int(box.cls)
                class_names = model.names

                if isinstance(class_names, dict):
                    detected_class_name = class_names.get(class_id)
                else:
                    detected_class_name = class_names[class_id] if class_id < len(class_names) else None

                if detected_class_name == HANGER_CLASS_NAME:
                    hanger_boxes.append(box)

        output_img = r.plot()

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        result_filename = f"result_{timestamp_str}.jpg"
        result_filepath = os.path.join("static", result_filename)
        os.makedirs("static", exist_ok=True)
        cv2.imwrite(result_filepath, output_img)
        result_url = f"/static/{result_filename}"

        request_data = {
            "timestamp": datetime.now().isoformat(),
            "input_filename": image.filename or "unknown",
            "found_hangers": len(hanger_boxes),
            "result_image_url": result_url,
            "total_objects_detected": total_objects_detected
        }
        history.append(request_data)

        return {
            "count": len(hanger_boxes),
            "result_image_url": result_url,
            "total_objects_detected": total_objects_detected,
            "message": f"Обработка завершена. Найдено вешалок: {len(hanger_boxes)}."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Critical error in analyze_wardrobe: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


history = []


@app.get("/history")
async def get_history():
    return {"history": history}


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/test")
async def test_route():
    return {"status": "ok", "message": "Server is running"}
