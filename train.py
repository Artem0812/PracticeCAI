from ultralytics import YOLO

def main():
    # Загружаем модель
    model = YOLO("yolov8n.pt")

    # Путь такой:
    data_yaml_path = "dataset/hanger.v1i.yolov8/data.yaml"

    print(f"Используем конфиг датасета: {data_yaml_path}")

    # Запускаем обучение
    results = model.train(
        data=data_yaml_path,     # Путь к data.yaml
        epochs=200,              # Количество эпох
        imgsz=640,               # Размер изображения
        batch=-1,                # Размер батча
        name="train_run-final",  # Имя запуска
        project="hanger_detection", # Папка проекта внутри runs/
        mosaic=1.0,              # Склеивает картинки в одну, помогая учиться на малом количестве данных
        mixup=0.5,               # Смешивание изображений
        device="cpu",            # Используем CPU
        val=False,               # Отключает валидацию
        verbose=True,            # Подробный вывод логов
        plots=True               # Сохранять графики обучения (precision, recall и т.п.)
    )

    # Вывод результатов
    print("Обучение завершено!")
    print(results)

if __name__ == "__main__":
    main()
