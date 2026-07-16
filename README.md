# 📊 Predictive Student Analytics Dashboard

An end-to-end desktop GUI application built in Python that uses Machine Learning to predict student final marks based on daily behavioral data.

## 🚀 Core Features
* **Interactive Desktop GUI:** Built with Python's graphical interface tools for real-time live testing.
* **Production-Ready Models:** Implements optimized Random Forest and XGBoost regressions.
* **Leakage-Free Pipeline:** Engineered with a strict train-test split validation framework to prevent overfitting.

## 📈 Model Performance Profiles
Our evaluation loop utilizes a strict 80/20 train-test validation split on unseen data:
* **XGBoost / Random Forest R² Score:** 0.90+ (Highly stable variance explanation)
* **Mean Absolute Error (MAE):** ~2.3 Marks (Average prediction deviation)

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn, XGBoost, NumPy, Pandas
* **Interface:** Tkinter / CustomTkinter

## 📋 Features Analyzed
1. Study Hours
2. Attendance Rate (%)
3. Sleep Hours
4. Previous Marks
5. Assignment Scores
6. Participation Rate

## 💻 How To Run

### 1. Clone the repository

```bash
git clone https://github.com/rumana-mca/student-performance-predictor.git
```

### 2. Move into the project folder

```bash
cd student-performance-predictor
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python student_gui.py
```
## 🚀 Future Improvements

- AI-generated personalized recommendations
- Cloud deployment for remote accessibility
- Teacher dashboard with class-wise analytics
- Student progress tracking over multiple semesters
- Email notifications for at-risk students
- Multi-user authentication and role-based access
