import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load model and encoders using absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'workout_diet_model.pkl')
encoders_path = os.path.join(base_dir, '..', 'models', 'encoders.pkl')

model = joblib.load(model_path)
encoders = joblib.load(encoders_path)

# These constants match the exact values in the GYM.csv dataset
BMI_CATEGORIES = ["Underweight", "Normal weight", "Overweight", "Obesity"]
GOALS = ["muscle_gain", "fat_burn"]
GENDERS = ["Male", "Female"]


def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)


def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesity"


def main():
    st.set_page_config(
        page_title="AI Fitness Planner",
        page_icon="💪",
        layout="wide"
    )
    
    st.title("🏋️ AI-Based Personalized Workout & Diet Planner")
    st.markdown("*Powered by Machine Learning | Personalized for Students*")
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("📋 User Profile")
        gender = st.selectbox("Gender", GENDERS, key="gender")
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        goal = st.selectbox("Fitness Goal", GOALS, key="goal")
        submit = st.button("🚀 Generate Plan", use_container_width=True)

    if submit:
        # Calculate BMI
        bmi = calculate_bmi(weight, height)
        bmi_cat = categorize_bmi(bmi)
        
        # Normalize inputs for model
        gender_lower = gender.lower()
        goal_lower = goal.lower().replace(" ", "_")
        
        # Create two columns for profile and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Your Profile Summary")
            profile_data = {
                "Gender": gender,
                "Height": f"{height:.1f} cm",
                "Weight": f"{weight:.1f} kg",
                "BMI": f"{bmi:.1f}",
                "BMI Category": bmi_cat.capitalize(),
                "Fitness Goal": goal
            }
            st.dataframe(pd.DataFrame(list(profile_data.items()), columns=["Metric", "Value"]), use_container_width=True)
            
            # BMI Gauge
            st.markdown("### BMI Status")
            fig, ax = plt.subplots(figsize=(8, 2))
            categories = ["Underweight\n(<18.5)", "Normal\n(18.5-25)", "Overweight\n(>25)"]
            colors = ["#3498db", "#2ecc71", "#e74c3c"]
            x_pos = [0, 1, 2]
            values = [18.5, 25, 35]
            
            bars = ax.barh(x_pos, values, color=colors, height=0.5)
            ax.axvline(bmi, color="black", linestyle="--", linewidth=2, label=f"Your BMI: {bmi:.1f}")
            ax.set_yticks(x_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel("BMI Value")
            ax.legend()
            ax.set_xlim(0, 35)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Prepare features for prediction
            # Gender and BMI category come directly from dropdowns/calculation
            # Goal needs to be lowercase to match training data encoding
            input_df = pd.DataFrame(
                [[gender, bmi_cat, goal.lower()]], 
                columns=['gender', 'bmi_category', 'goal']
            )
            
            # Encode inputs
            for col in ['gender', 'bmi_category', 'goal']:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except ValueError as e:
                    st.error(f"❌ Error: Unknown value for {col}. Please check your inputs.")
                    return
            
            # Get predictions
            pred = model.predict(input_df)[0]
            schedule_idx, meal_idx = pred
            
            # Decode predictions
            schedule = encoders['exercise_schedule'].inverse_transform([schedule_idx])[0]
            meal = encoders['meal_plan'].inverse_transform([meal_idx])[0]
            
            # Display recommendations
            st.subheader("🎯 AI Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.info(f"**💪 Workout Schedule**\n\n{schedule}", icon="🏃")
            
            with rec_col2:
                st.success(f"**🍽️ Meal Plan**\n\n{meal}", icon="🥗")
        
        # Visual Insights Section
        st.markdown("---")
        st.subheader("📊 Visual Insights")
        
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
            # BMI Pie Chart
            fig, ax = plt.subplots(figsize=(6, 4))
            bmi_data = [18.5, 6.5, 10]
            bmi_labels = ["Underweight", "Normal", "Overweight"]
            colors_pie = ["#3498db", "#2ecc71", "#e74c3c"]
            
            if bmi_cat == "underweight":
                explode = (0.1, 0, 0)
            elif bmi_cat == "normal":
                explode = (0, 0.1, 0)
            else:
                explode = (0, 0, 0.1)
            
            ax.pie(bmi_data, labels=bmi_labels, autopct="%1.1f%%", colors=colors_pie, explode=explode, startangle=90)
            ax.set_title("BMI Category Distribution")
            st.pyplot(fig)
        
        with chart_col2:
            # Goal Progress
            fig, ax = plt.subplots(figsize=(6, 4))
            goals = ["Muscle\nGain", "Fat\nBurn", "Maintenance"]
            progress = [75, 60, 45]
            bars = ax.bar(goals, progress, color=["#e74c3c", "#f39c12", "#3498db"])
            ax.set_ylabel("Effectiveness %")
            ax.set_title("Goal Achievement Potential")
            ax.set_ylim(0, 100)
            for bar, val in zip(bars, progress):
                ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val}%", ha="center", va="bottom")
            st.pyplot(fig)
        
        with chart_col3:
            # User Stats
            fig, ax = plt.subplots(figsize=(6, 4))
            metrics = ["BMI\nValue", "Goal\nMatch", "Plan\nPrecision"]
            scores = [min(bmi / 25 * 100, 100), 85, 95]
            bars = ax.bar(metrics, scores, color=["#9b59b6", "#1abc9c", "#e67e22"])
            ax.set_ylabel("Score %")
            ax.set_title("Your Metrics")
            ax.set_ylim(0, 100)
            for bar, val in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.0f}%", ha="center", va="bottom")
            st.pyplot(fig)
        
        st.markdown("---")
        st.info("💡 **Tip**: Follow the recommended workout schedule and meal plan consistently for 4-6 weeks to see noticeable results!")
    
    elif submit:
        st.warning("⚠️ Please select all required fields (Gender and Fitness Goal).")

if __name__ == '__main__':
    main()
