# 🌽 Corn Emergence Estimation from UAV Imagery

<p align="center">
  <img src="https://media.giphy.com/media/ZZrP0uQKFF5yA/giphy.gif" width="600"/>
</p>

Автоматическая оценка всхожести кукурузы на основе изображений с БПЛА. Проект включает два модуля:

- **Сегментация растений** с использованием глубоких сверточных сетей (`UNet`, `DeepLab`)
- **Извлечение признаков и классификация всхожести** с помощью PySpark и ML

---

## 📂 Структура проекта

```bash
Maize/
├── Classification/              # Модуль оценки всхожести
│   ├── ALE_PREDICT_*            # Предсказания на полях Алейска
│   ├── POS_PREDICT_*            # Предсказания на полях Поспелихи
│   ├── Dataset/                 # Данные
│   ├── config.py                # Конфигурации
│   ├── feature_engineering.py   # PySpark обработка признаков
│   ├── hough.py                 # Выделение линий роста
│   ├── inference_utils.py       # Вспомогательные функции инференса
│   ├── Inference.py             # Скрипт инференса
│   ├── spark_session.py         # Инициализация Spark
│   ├── visualization.py         # Визуализации
│   └── ...                      # Остальные утилиты
│
├── segmentation_project/        # Модуль сегментации
│   ├── Config.py
│   ├── Dataset.py
│   ├── Inference.py
│   ├── Model.py
│   ├── Train.py
│   ├── Visualization.py
│   └── ...
│ 
├── Web-Interface/        # Модуль веб-сервиса
│   ├── load_models.py
│   ├── main.py
│   ├── Streamlit_app.py
│   └── ...
│
├── data_lee.csv                 # Основной CSV с аннотациями
└── README.md                    # Этот файл

