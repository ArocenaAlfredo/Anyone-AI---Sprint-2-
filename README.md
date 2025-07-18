# Credit Risk Classification

This project addresses a binary classification problem: predicting whether an applicant for a home credit loan will default on payments. The goal is to support financial institutions in making informed lending decisions using machine learning.

## 🧠 Problem Statement

Given structured customer data, the model predicts a target variable:

- `1` if the client is likely to have payment difficulties.
- `0` otherwise.

The project uses **Area Under the ROC Curve (AUC-ROC)** as the evaluation metric, with the model returning the probability of default for each applicant.

## 📊 Dataset

The dataset includes two main files:

- `application_train_aai.csv`: labeled training data.
- `application_test_aai.csv`: unlabeled test data.

These files contain structured features such as demographics, financial information, and loan details.

## 🛠️ Technologies Used

- **Python**: Main programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Model training and evaluation
- **Matplotlib / Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development
- **Black + isort**: Code formatting
- **Pytest**: Unit testing

## 🧪 Project Structure

.
├── src/ # Core functions and ML code
├── notebooks/ # Jupyter notebooks with analysis
├── tests/ # Unit tests
├── requirements.txt
└── README.md

bash
Copiar
Editar

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
🧪 Run Tests
bash
Copiar
Editar
pytest tests/
🧼 Code Formatting
To ensure consistent code style:

bash
Copiar
Editar
isort --profile=black .
black --line-length 88 .
📈 Model Output
The model outputs probability scores of loan default and is evaluated based on AUC-ROC. Visualizations and performance metrics are included in the main notebook.

This project simulates a real-world financial modeling use case and demonstrates a complete ML pipeline from preprocessing to evaluation.
