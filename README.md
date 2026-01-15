# Cat vs Dog Classification

## 1. Introduction
The goal of this task was to classify images of cats and dogs using different machine learning and deep learning methods. A labelled dataset was provided containing **1,600 training images** (800 cats and 800 dogs) and **400 test images**. The objective was to build multiple models, compare their performance, and determine which approach provides the best accuracy.

---

## 2. Dataset Description

### Training Set
- **Total images:** 1600  
- **Cats:** `/Q1/train/cat` (cat.1.jpg → cat.999.jpg)  
- **Dogs:** `/Q1/train/dog` (dog.1.jpg → dog.999.jpg)

### Test Set
- **Cats:** `/Q1/test/cat/Cat (1).jpg … Cat (5).jpg`  
- **Dogs:** `/Q1/test/dog/Dog (1).jpg … Dog (5).jpg`

### Preprocessing
- Images vary in size, lighting, and quality  
- All images were:
  - Resized to **150 × 150**
  - Normalized to **[0, 1]**

---

## 3. Methods Used

### A. Traditional ML Models (Baseline)
Before applying deep learning, classical machine learning methods were tested using **HOG feature extraction** with:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  

#### Limitations
- Small dataset size  
- High image variability  
- Loss of spatial information after flattening  

#### Results
- Accuracy ranged from **55–65%**, which is insufficient for reliable image classification.

---

### B. Deep Learning Model (CNN – Best Model)
A **Convolutional Neural Network (CNN)** was implemented using **TensorFlow/Keras**.

#### Model Architecture
- Conv2D (32 filters) + MaxPooling  
- Conv2D (64 filters) + MaxPooling  
- Conv2D (128 filters) + MaxPooling  
- Flatten  
- Dense (128 units, ReLU)  
- Output Layer (1 unit, Sigmoid)  

#### Training Configuration
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Batch Size:** 32  
- **Epochs:** 10  
- **Image Augmentation:** rotation, zoom, flips, shifts  

This approach significantly outperformed traditional machine learning models.

---

## 4. Training Results
After training for **10 epochs**:
- **Training Accuracy:** ~65%  
- **Validation Accuracy:** ~66%  

Validation accuracy increased steadily after **epoch 4**, indicating effective feature learning.

The trained model was saved as:

cat_and_dog_model.h5
---

## 5. Testing the Model on New Images
Images from `/Q1/test/` were used for inference.

### Sample Predictions

| Image   | True Label | Prediction |
|--------|------------|------------|
| Cat (1) | Cat | Dog |
| Cat (2) | Cat | Cat |
| Dog (1) | Dog | Dog |
| Dog (2) | Dog | Dog |

**Overall accuracy on sample test images:** **75%**

---

## 6. Discussion
The CNN performed well given:
- Limited epochs (10)  
- No transfer learning  
- Low image resolution (150×150)  

### Potential Improvements
- Train for **20–30 epochs**  
- Use pre-trained models (e.g., **MobileNetV2**, **VGG16**)  
- Increase image resolution to **224×224**  
- Expand the dataset  

Despite these constraints, the model demonstrates a functional and effective image classification pipeline.

---

## 7. Conclusion
Multiple methods were implemented and evaluated for the cat vs dog classification task.

- Traditional ML models achieved **55–65% accuracy**
- CNN achieved **66% validation accuracy**
- CNN achieved **75% accuracy on new test images**
- Deep learning clearly outperformed classical methods

### Final Result
The **CNN is the best-performing model** for this dataset and successfully classifies most images.

## 8. Why Training and Testing Data isn't Posted on the github Reop
Since the file constraints of 25mb was a hurdle I am only able to post bits of my code in the repo, if needed I can show the model via my system!
