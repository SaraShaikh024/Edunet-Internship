# AI-Based Personalized Workout & Diet Planner for Students

This project is an AI-powered recommendation system that generates personalized workout schedules and meal plans for students based on their gender, fitness goal, and BMI category.

## Features

- User inputs gender, height, weight, and fitness goal
- System calculates BMI and categorizes it
- Machine learning model (Decision Tree/Random Forest) predicts exercise schedule and meal plan
- Dashboard displays profile, recommendations, and visual insights
- Optional progress tracking

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

- Train the model: `python src/train_model.py`
- Run the dashboard: `streamlit run src/app.py`
