# 🥔 Potato Disease Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify potato leaf images into three categories:

- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

The model is trained using TensorFlow and Keras on the PlantVillage dataset.

---

## 📂 Dataset

- **Total Images:** 2152
- **Classes:** Early Blight, Late Blight, Healthy
- **Image Size:** 256x256
- **Splits:**
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

---

## 🧠 Model Architecture

```python
Sequential([
    Resizing(256, 256),
    Rescaling(1./255),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3)  # 3 output classes
])
```

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy
- **EarlyStopping:** Enabled to prevent overfitting

---

## 📊 Evaluation

- Accuracy and loss are tracked for both training and validation sets.
- A sample batch of predictions (9 images) is visualized along with their predicted and actual labels.

---

## 🚀 How to Run

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure the dataset folder path is:**
   ```
   ../PlantVillagePotatoDataset/
   ```
3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook model.ipynb
   ```

---

## 📝 Output Sample

Displays predictions with confidence scores and actual labels for 9 sample test images.

---

## 📌 .gitignore Suggestions

- Ignore model files, datasets, cache, logs, and system files to keep the repository clean.

---

# 🌐 Web App: Streamlit Interface

A user-friendly web application is provided for potato disease classification using Streamlit.

### Features
- Upload a potato leaf image (JPG, JPEG, PNG)
- The app displays the uploaded image and predicts its condition: Healthy, Early Blight, or Late Blight
- Shows the prediction confidence score
- Clean, centered UI with clear result display

### How to Run the Web App

1. Make sure you have all dependencies installed (see requirements.txt)
2. Ensure the trained model file is available at: `../models/potato_disease_classification_v1.0.h5`
3. From the `server/` directory, run:
   ```bash
   streamlit run server.py
   ```
4. Open the provided local URL in your browser to use the app.

### App UI Overview
- The main page allows you to upload a potato leaf image.
- The left panel shows the input image.
- The right panel displays the predicted disease class and confidence.
- Supported classes: Early Blight, Late Blight, Healthy

---

## 🔮 Future Development

- **XAI Integration:**
  - Plan to integrate Explainable AI (XAI) techniques (such as Grad-CAM or LIME) to provide visual explanations for model predictions, helping users and researchers understand which parts of the leaf image influenced the classification decision.
- Additional improvements and features will be considered based on user feedback and research advancements.

