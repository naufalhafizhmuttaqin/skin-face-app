# 🧴 Face Skin Type Detection

A web application that predicts a person's facial skin type from a photo using a deep learning model (InceptionV3). Users simply upload a photo or take one with their camera, and the app returns the predicted skin type along with a confidence score.

**🔗 Live demo:** https://YOUR-APP-NAME.streamlit.app

## Supported skin types

| Class | Description |
|---|---|
| `berjerawat` | Acne-prone skin |
| `berminyak` | Oily skin |
| `kering` | Dry skin |
| `normal` | Normal skin |

## Features

- Upload a photo (JPG/PNG) or capture one directly with the camera
- Probability breakdown for every class
- Automatic low-confidence warning (when the top prediction is below 50%)
- Uploaded photos are not stored by the app

## Model

- **Architecture:** InceptionV3 (ImageNet pre-trained, truncated at the `mixed4` layer) + GlobalAveragePooling, BatchNormalization, Dense (256), Dropout (0.5), Dense (4, softmax)
- **Input:** 299 × 299 pixels, normalized with `/255`
- **Training:** transfer learning (100 epochs) followed by fine-tuning (25 epochs)
- **Dataset:** [Face Skin Type](https://www.kaggle.com/datasets/muttaqin1113/face-skin-type), 1,735 images split into 70% train / 20% validation / 10% test

Results on the validation set:

| Metric | Score |
|---|---|
| Accuracy | 0.90 |
| Macro F1-score | 0.90 |

> **Note:** Performance on photos outside the training dataset (for example, selfies taken with different lighting or cameras) may be lower than the figures above.

## Project structure

```
.
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── inceptionv3_faceskin.keras  # trained model
└── README.md
```

## Run locally

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Tips for better results

- Use a close-up face photo with good, even lighting
- Avoid filters, heavy makeup, and blurry images
- Make sure the face is clearly visible and unobstructed

## Disclaimer

This application provides an estimate based on an AI model and is **not a substitute for diagnosis or advice from a dermatologist**. For any serious skin concern, please consult a qualified medical professional.

## Built with

Python · TensorFlow/Keras · Streamlit · Altair · Pillow
