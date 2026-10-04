# Text Sentiment Analysis using RNN & LSTM

## 📌 Project Description

This project performs **Text Sentiment Analysis** using a **Long Short-Term Memory (LSTM)** neural network.

The **IMDB Movie Review Dataset** is used to classify movie reviews into:

* **Positive**
* **Negative**

The project uses an **Embedding Layer**, **LSTM Layer**, and **Dense Layer** to understand the sequence and sentiment of text reviews.

---

## 🧠 Model Architecture

```text
Movie Review
     ↓
IMDB Dataset
     ↓
Padding
     ↓
Embedding Layer
     ↓
LSTM Layer
     ↓
Dense Layer
     ↓
Sigmoid
     ↓
Positive / Negative
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* RNN
* LSTM
* Deep Learning
* Natural Language Processing (NLP)

---

## 📂 Project Structure

```text
Text_Sentiment_Analysis_RNN_LSTM/
│
├── TextSentimentAnalysis.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── Screenshots/
    ├── training_output.png
    └── prediction_output.png
```

---

## ⚙️ Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Python file:

```bash
python TextSentimentAnalysis.py
```

The IMDB dataset will be loaded automatically using TensorFlow/Keras.

---

## 📊 Dataset

The project uses the **IMDB Movie Review Dataset** provided by TensorFlow/Keras.

* Training Reviews: 25,000
* Testing Reviews: 25,000
* Vocabulary Size: 10,000
* Maximum Review Length: 200 words

### Sentiment Labels

```text
0 → Negative
1 → Positive
```

---

## 🔍 Prediction

The trained LSTM model predicts the sentiment probability of a test review.

```text
Probability >= 0.5 → POSITIVE
Probability < 0.5  → NEGATIVE
```

---

## 📸 Sample Output

```text
----------------------------------------
Final Result
----------------------------------------
Prediction Probability : 0.91
Actual Sentiment       : POSITIVE
Predicted Sentiment    : POSITIVE
----------------------------------------
```

---

## 🚀 Future Improvements

* Add user-input review prediction
* Improve model accuracy
* Add Bidirectional LSTM
* Add GRU model
* Create a web interface using Flask or Streamlit

---

## 👨‍💻 Author

**Yash Chavan**

### Project Name

**Text_Sentiment_Analysis_RNN_LSTM**

### Date

**04/10/2026**
