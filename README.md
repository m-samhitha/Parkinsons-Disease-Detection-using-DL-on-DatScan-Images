# Parkinsons-Disease-Detection-using-DL-on-DatScan-Images

This project implements a deep learning model for early detection of Parkinson's Disease using DaTScan brain imaging data.
The model analyzes dopamine transporter activity patterns in brain scans to classify patients as Parkinson’s or Healthy.


## 🚀 Features

* Uses deep learning models for medical image classification
* Applies image preprocessing and normalization
* Implements data augmentation to improve performance
* Generates confusion matrix and classification report
* Visualizes training and validation accuracy curves


## 📂 Dataset

Dataset: *(Add your Google Drive link here)*

The dataset contains:

* DaTScan brain images
* Two classes:

  * Parkinson’s
  * Healthy

### Dataset Structure

<img width="380" height="531" alt="image" src="https://github.com/user-attachments/assets/48a5f975-d37b-4562-82cb-cbaaed4e1198" />

## 🛠️ Requirements

* Python 3.x
* TensorFlow / Keras
* OpenCV
* matplotlib
* scikit-learn
* seaborn
* Google Colab (recommended)

Install dependencies using:

```
pip install tensorflow opencv-python matplotlib scikit-learn seaborn
```


## ⚙️ How to Run

1. **Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Set Dataset Path**

```python
base_dir = "/content/drive/MyDrive/DaTScan_dataset"
```

3. **Train the Model**

```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks
)
```

4. **Fine-Tune the Model**

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

history_finetune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=callbacks
)
```

5. **Evaluate the Model**

```python
loss, acc = model.evaluate(test_generator)
print(f"Final Test Accuracy: {acc*100:.2f}%")
```

---

## 🏗️ Model Architecture

* Base Model: Transfer Learning (DenseNet / EfficientNet / VGG16)
* Pooling Layer: GlobalAveragePooling2D
* Fully Connected Layer: Dense with ReLU
* Dropout Layers for regularization
* Output Layer: Sigmoid (Binary Classification)
* Optimizer: Adam
* Loss Function: Binary Crossentropy


## 📊 Evaluation

* Accuracy curves (Training vs Validation)
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)


## 👩‍💻 Author

**Samhitha Moparthi**
📧 [samhithamoparthi@gmail.com](mailto:samhithamoparthi@gmail.com)

**Gayathri Mocharla**
📧 [mocharlagayathri@gmail.com](mailto:mocharlagayathri@gmail.com)


## ⚠️ Note

This project is for academic and research purposes only and not intended for medical diagnosis.
