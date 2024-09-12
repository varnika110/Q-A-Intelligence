# Question-Answering Model 🤖

## 🚀 Overview
This repository contains a sophisticated question-answering model that leverages advanced NLP techniques. The pipeline encompasses preprocessing, feature engineering, model training, and evaluation. The model utilizes embeddings from **DistilBERT** and **Universal Sentence Encoder** to achieve high performance in handling question-answering tasks.

## 📦 Dependencies
Ensure the following libraries are installed:

- **transformers**: For model and tokenizer
- **torch**: For tensor operations
- **numpy** and **pandas**: For data manipulation
- **random**, **string**, **datetime**: For utility functions
- **nltk**: For natural language processing
- **tensorflow** and **tensorflow_hub**: For embeddings
- **scikit-learn**: For machine learning utilities
- **scipy**: For statistical functions

## 🛠 Setup and Configuration
### 1. Environment Setup
Clone this repository and set up a Python environment with the required dependencies:

 ```bash
git clone https://github.com/varnika110/Q-A-Intelligence.git
cd question-answering-model
 ```

### 2. Download Pre-trained Models
Download the pre-trained models for DistilBERT and Universal Sentence Encoder and place them in the designated directories:

- **DistilBERT:** ../input/distilbertbaseuncased/
- **Universal Sentence Encoder:** ../input/universalsentenceencoderlarge4/

## 🔧 Usage

### 1. Data Preprocessing
The preprocess_data function prepares the training and test datasets by:

- Calculating text statistics (length, number of stopwords, etc.)
- Feature engineering for domain information
- Creating features based on word overlap

### 2. Embedding Computation
The compute_embeddings function extracts sentence embeddings for questions and answers using:

- **DistilBERT** for token-level embeddings
- **Universal** Sentence Encoder for sentence-level embeddings

### 3. Model Training and Evaluation
The train_and_evaluate function trains a Ridge regression model on the extracted features and evaluates its performance using the Spearman correlation coefficient.

## 📈 Performance
The model's effectiveness is measured using the Spearman correlation coefficient, assessing how well the predicted answers align with the true answers.

## 📜 Code Explanation
- **fetch_vectors:** Generates embeddings using DistilBERT.
- **chunks:** Splits data into manageable chunks.
- **spearman_corr:** Computes Spearman correlation for evaluation.
- **preprocess_data:** Manages all preprocessing and feature engineering tasks.
- **compute_embeddings:** Obtains embeddings for textual data.
- **train_and_evaluate:** Trains the model and computes performance metrics.

## 📝 Notes
Ensure data files are correctly placed and paths are updated.
For large datasets, adjust batch sizes and memory usage as necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 📜
