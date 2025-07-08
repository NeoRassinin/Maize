# Maize: Комплексный пайплайн для анализа кукурузы

Этот репозиторий содержит полную архитектуру проекта для анализа изображений кукурузы, включая:
- `segmentation/` — Сегментация изображений кукурузы
- `classification/` — Извлечение признаков на основе сегментации и скелетизации и обучение моделей
- `web_interface/` — веб-интерфейс для работы с пайплайном и отображения результатов

---

## Структура проекта

```
Maize/
├── segmentation/             # Сегментация и анализ
│   ├── Config.py
│   ├── Preprocessing.py
│   ├── Dataset.py
│   ├── Inference.py
│   ├── Metrics.py
│   ├── Model.py
│   ├── Train.py         
│   └── Visualization.py
│
├── classification/          # Модели и пайплайн классификации
│   ├── Config.py
│   ├── Feature_Engineering.py
│   ├── Hough.py
│   ├── Inference.py
│   ├── Model.py
│   └── Preprocessing.py         
│
├── web_interface/           # Flask-интерфейс для запуска и визуализации
│   ├── load_models.py
│   ├── main.py
│   └── Stramlit_app.py
│
├── requirements.txt         # Зависимости
└── README.md
```


## 🌐 web_interface/

Интерактивный веб-интерфейс на Streamlit:
- Загрузка изображения
- Автоматическая сегментация + классификация
- Отображение результатов

### Запуск интерфейса:
```bash
cd Web_interface
python Streamlit_app.py
```

После запуска доступно на вашем локальном хосте

---

## 🛠 Установка зависимостей

```bash
pip install -r requirements.txt
```

---



---

## 👨‍💻 Автор
[NeoRassinin](https://github.com/NeoRassinin)

---
