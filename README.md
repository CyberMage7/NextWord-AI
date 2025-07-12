# NextWord-AI

A sophisticated next-word prediction system using LSTM neural networks, trained on Shakespeare's Hamlet. This project demonstrates advanced NLP techniques with deep learning for text generation and word prediction.

## 🚀 Features

- **LSTM-based Neural Network**: Advanced recurrent neural network architecture for sequence prediction
- **Shakespeare's Hamlet Dataset**: Trained on classic literature for rich linguistic patterns
- **Early Stopping**: Prevents overfitting with intelligent training termination
- **Interactive Web Interface**: Streamlit-powered UI for real-time predictions
- **Model Persistence**: Save and load trained models for future use
- **Preprocessing Pipeline**: Complete text tokenization and sequence preparation

## 📁 Project Structure

```
NextWord-AI/
├── app.py                 # Streamlit web application
├── experiments.ipynb      # Jupyter notebook with model development
├── next_word_lstm.h5     # Trained LSTM model
├── tokenizer.pickle      # Saved tokenizer for text processing
├── hamlet.txt           # Shakespeare's Hamlet text dataset
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/CyberMage7/NextWord-AI.git
   cd NextWord-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Running the Web Application

Launch the Streamlit interface for interactive predictions:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser and enter text to get next-word predictions.

### Training Your Own Model

Open and run the `experiments.ipynb` notebook to:
- Download and preprocess the Hamlet dataset
- Train the LSTM model with early stopping
- Save the trained model and tokenizer
- Test predictions on custom text

## 🧠 Model Architecture

The LSTM model features:
- **Embedding Layer**: 100-dimensional word embeddings
- **LSTM Layers**: Two LSTM layers (150 and 100 units) with dropout
- **Output Layer**: Softmax activation for word probability distribution
- **Early Stopping**: Monitors validation loss with patience of 5 epochs

## 📊 Training Process

1. **Data Collection**: Downloads Shakespeare's Hamlet from NLTK corpus
2. **Preprocessing**: Tokenizes text and creates n-gram sequences
3. **Sequence Padding**: Ensures uniform input length
4. **Train/Test Split**: 80/20 split for model validation
5. **Model Training**: Uses categorical crossentropy loss with Adam optimizer

## 🎨 Example Predictions

```
Input: "To be or not to"
Prediction: "be"

Input: "To be bad is better than"
Prediction: [context-dependent prediction]
```

## 🔧 Technical Details

- **Framework**: TensorFlow/Keras
- **Architecture**: Sequential LSTM
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Validation**: Early stopping with best weight restoration

## 📦 Dependencies

- tensorflow>=2.16.0
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorboard
- streamlit
- scikeras
- nltk

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- Shakespeare's works via NLTK Gutenberg corpus
- TensorFlow team for the deep learning framework
- Streamlit for the web interface framework