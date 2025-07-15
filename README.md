🥔 Potato Disease Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify potato leaf images into three categories:

- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

The model is trained using TensorFlow and Keras on the PlantVillage dataset.

📂 Dataset

- Total Images: 2152
- Classes: Early Blight, Late Blight, Healthy
- Image Size: 256x256
- Splits:
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

🧠 Model Architecture

```
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

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy
- EarlyStopping: Enabled to prevent overfitting

📊 Evaluation

- Accuracy and loss are tracked for both training and validation sets.
- A sample batch of predictions (9 images) is visualized along with their predicted and actual labels.

🚀 How to Run

Clone the repository and install dependencies:

```
pip install -r requirements.txt
```

Ensure the dataset folder path is:

```
../PlantVillagePotatoDataset/
```

Run the Jupyter notebook:

```
jupyter notebook model.ipynb
```

📝 Output Sample

Displays predictions with confidence scores and actual labels for 9 sample test images.

📌 .gitignore Suggestions

```

---

🌐 Web App: Streamlit Interface

A user-friendly web application is provided for potato disease classification using Streamlit.

**Features:**
- Upload a potato leaf image (JPG, JPEG, PNG)
- The app displays the uploaded image and predicts its condition: Healthy, Early Blight, or Late Blight
- Shows the prediction confidence score
- Clean, centered UI with clear result display

**How to Run the Web App:**

1. Make sure you have all dependencies installed (see requirements.txt)
2. Ensure the trained model file is available at: `../models/potato_disease_classification_v1.0.h5`
3. From the `server/` directory, run:

   ```bash
   streamlit run server.py
   ```

4. Open the provided local URL in your browser to use the app.

**App UI Overview:**
- The main page allows you to upload a potato leaf image.
- The left panel shows the input image.
- The right panel displays the predicted disease class and confidence.
- Supported classes: Early Blight, Late Blight, Healthy



---

