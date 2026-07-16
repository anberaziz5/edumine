# EduMine - Early Warning System for Student Success

**Live Demo:** [https://edumine-sage.vercel.app/](https://edumine-sage.vercel.app/)

EduMine is a machine-learning-powered early warning system designed to predict the likelihood of student dropout based on early engagement metrics in online learning environments. By leveraging data-driven insights, EduMine empowers educators to identify at-risk students before they fall behind and intervene effectively.

## 🚀 Features

- **Data-Driven Insights:** Trained on over 32,000 records from the Open University Learning Analytics Dataset (OULAD), the model recognizes complex patterns in student behavior.
- **Early Intervention:** Analyzes engagement from just the first few weeks of a course to provide early warnings, giving educators time to offer meaningful support.
- **Explainable AI (SHAP):** Doesn't just provide a risk score; uses SHAP (SHapley Additive exPlanations) to highlight exactly which factors are increasing or reducing a student's risk.
- **Beautiful & Intuitive UI:** Built with React and Tailwind CSS, featuring a premium glassmorphic design and engaging micro-animations.

## 🏗️ Architecture

The project is structured into three main components:

- **Frontend (`/frontend`)**: A React application built with Vite and Tailwind CSS.
- **Backend API (`/api`)**: A FastAPI Python application serving the machine learning model.
- **Model Training (`/model`)**: Scripts and logic for training the XGBoost predictive model.

## 🛠️ Tech Stack

- **Frontend:** React, Vite, Tailwind CSS
- **Backend:** FastAPI, Python, Uvicorn
- **Machine Learning:** XGBoost, SHAP, Scikit-Learn
- **Deployment:** Vercel (Frontend)

## 🏃‍♂️ Getting Started (Local Development)

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

### Backend Setup

1. Navigate to the API directory:
   ```bash
   cd api
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python main.py
   # or
   uvicorn main:app --reload
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
