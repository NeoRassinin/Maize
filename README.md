## 🌽 About The Project

This project provides a **modular ML pipeline** to assess the germination potential of corn (maize) kernels. It includes image segmentation using U-Net, feature extraction (skeleton, contour, bounding boxes), and classification using various ML algorithms (Random Forest, SHAP explainability, etc).

###  Why This Project

* Automated seedling evaluation using computer vision
* Designed for high-throughput phenotyping tasks in agriculture
* Flexible structure to support plug-and-play segmentation and classification models

##  Project Structure

```
Maize/
├── Classification/              # Germination assessment module
│   ├── ALE_PREDICT_*            # Predictions in the Fields of Aleysk
│   ├── POS_PREDICT_*            # Predictions in the Fields of Pospelikha
│   ├── Dataset/                 # Data
│   ├── config.py                # Configurations
│   ├── feature_engineering.py   # PySpark feature processing
│   ├── hough.py                 # Highlighting growth lines
│   ├── inference_utils.py       # Auxiliary functions of the inference
│   ├── Inference.py             # The Inference script
│   ├── spark_session.py         # Spark Initialization
│   ├── visualization.py         # Visualizations
│   └── ...                      # Other utilities
│
├── segmentation_project/        # The segmentation module
│   ├── Config.py                # Configurations
│   ├── Dataset.py               # Data
│   ├── Inference.py             # The Inference script
│   ├── Model.py                 # Neural network architecture
│   ├── Train.py                 # Training pipeline for training a segmentation model
│   ├── Visualization.py         # Utilities for visualizing predictions, labels, masks, etc.
│   └── ...
│ 
├── Web-Interface/               # A web interface module on Streamlit to demonstrate a model
│   ├── load_models.py           # Downloading and preparing models for use in the web
│   ├── main.py                  # The main starting point for local interface testing
│   ├── Streamlit_app.py         # Streamlit application: UI for uploading images and displaying results
│   └── ...
│              
└── data_lee.csv     # Basic CSV with annotations        
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

##  Acknowledgments

* [PyImageSearch](https://pyimagesearch.com/)
* [Papers With Code](https://paperswithcode.com/)
* [Albumentations](https://albumentations.ai/)
* [Scikit-learn Docs](https://scikit-learn.org/stable/)

---



