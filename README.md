# Facial Recognition System

This repository contains a complete facial recognition system built from scratch using TensorFlow and OpenCV. The project includes:

- **Face Detection:** Utilizes OpenCV's Haar Cascade to detect faces in images.
- **Face Recognition:** Implements a Convolutional Neural Network (CNN) trained on the LFW (Labeled Faces in the Wild) dataset to classify detected faces by identity.

## Features

- **Detection and Recognition Pipeline:** Detect multiple faces in an image and recognize each face.
- **Training from Scratch:** The CNN is trained on the LFW dataset, which contains faces of various individuals.
- **Interactive Testing:** Upload an image through Google Colab and see detection and recognition results in real time.

## Requirements

- Python 3.x
- TensorFlow (Keras)
- OpenCV
- scikit-learn
- matplotlib
- Google Colab (recommended for ease of use)

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/facial-recognition-system.git
   cd facial-recognition-system

2. **Open the notebook:**

Open `facial_recognition_system.ipynb` in Google Colab.

3. **Run the notebook:**

Execute to train the model and test the face recognition system. You will be prompted to upload an image for testing.

## Project Structure
```
facial-recognition-system/
├── facial_recognition_system.ipynb  # Main Colab notebook
├── face_recognition_model.h5          # (Optional) Saved model after training
├── haarcascade_frontalface_default.xml # Haar Cascade file for face detection
└── README.md                          # Project documentation
```

## Notes
- The training process uses the LFW dataset, which might take several minutes to download and process.
- The recognition model in this example is for demonstration purposes. For production-level applications, consider using larger and more diverse datasets along with more advanced models.
- Contributions and suggestions are welcome!

## Acknowledgments
- [OpenCV](https://opencv.org/) for the Haar Cascade classifier.
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework.
- [LFW Dataset](https://datasets.activeloop.ai/docs/ml/datasets/lfw-dataset/) for providing a collection of labeled faces.
