## 🌽 About The Project

This project provides a **modular ML pipeline** to assess the germination potential of corn (maize) kernels. It includes image segmentation using U-Net, feature extraction (skeleton, contour, bounding boxes), and classification using various ML algorithms (Random Forest, SHAP explainability, etc).

###  Why This Project

* Automated seedling evaluation using computer vision
* Designed for high-throughput phenotyping tasks in agriculture
* Flexible structure to support plug-and-play segmentation and classification models

##  Project Structure

```
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
└── data_lee.csv     # Основной CSV с аннотациями           
```

##  Built With

* [Python](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyTorch](https://pytorch.org/)
* [Albumentations](https://albumentations.ai/)
* [SHAP](https://github.com/slundberg/shap)


## ⚡ Getting Started

Clone the project and install dependencies:

```bash
git clone https://github.com/your_username/Maize.git
cd Maize
pip install -r requirements.txt
```

Make sure `data_lee.csv` and image directories are properly structured under `Dataset/`.

### Run web-interface

```bash
python Web_Interface/main.py
```

### Run segmentation

```bash
python segmentation_project/Inference.py
```

### Run classification pipeline

```bash
python Classification/Inference.py
```


##  Pipeline Overview

1. **Segmentation**: U-Net model segments seed parts
2. **Feature Extraction**: Extracts contours, skeletons, bounding boxes, top-left points
3. **Classification**: Predicts germination using extracted features
4. **Visualization**: Overlays predictions and metrics on images

![Pipeline GIF](https://media.giphy.com/media/QBd2kLB5qDmysEXre9/giphy.gif)


##  Features

* [x] U-Net based image segmentation
* [x] Custom feature engineering (skeleton, contours)
* [x] Classification using Random Forest and SHAP
* [x] Configurable inference/visualization modules


##  Roadmap

* [ ] Add 3D seed visualization with Meshroom/Plotly
* [ ] Integrate model selection via Optuna
* [ ] Add real-time prediction UI


##  Contributing

Pull requests are welcome! Please follow these steps:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes
4. Push to the branch and open a PR


## 📄 License

Distributed under the MIT License. See `LICENSE` for more info.


## 💬 Contact

Project Author: **Rassinin Maxim**
Email: [youremail@example.com](mailto:youremail@example.com)
Repo: [github.com/your\_username/Maize](https://github.com/your_username/Maize)


##  Acknowledgments

* [PyImageSearch](https://pyimagesearch.com/)
* [Papers With Code](https://paperswithcode.com/)
* [Albumentations](https://albumentations.ai/)
* [Scikit-learn Docs](https://scikit-learn.org/stable/)

---



