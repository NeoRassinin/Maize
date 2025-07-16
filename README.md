# 🌽 Corn Emergence Estimation from UAV Imagery

<p align="center">
  <img src="https://media.giphy.com/media/ZZrP0uQKFF5yA/giphy.gif" width="600"/>
</p>

**🛰️ Проект по автоматической оценке всхожести кукурузы** с использованием снимков с БПЛА, методов сегментации, извлечения геометрических признаков и машинного обучения.

---

## 🔍 Цель

📊 Автоматизировать оценку всхожести по спутниковым/дроновым снимкам с точностью, сопоставимой с ручной агрономической экспертизой.

---

## 📁 Структура проекта

```bash
CornEmergence/
├── data/                # Сырые и обработанные изображения
├── notebooks/           # Jupyter-исследования
├── models/              # Обученные модели
├── segmentation/        # UNet, DeepLab и пр.
├── features/            # Геометрические признаки, скелетизация
├── spark_pipeline/      # Обработка через PySpark
├── inference/           # Скрипты предсказания
├── README.md
└── requirements.txt
