import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
# library for Gemini (Google AI Studio)
# Configure API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 👇 PUT MODEL HERE
model = genai.GenerativeModel("gemini-pro")

# Define your function below the model
def generate_meal_plan(prompt):
    response = model.generate_content(prompt)
    return response.text


# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================
st.set_page_config(
    page_title="FitGenius - AI Fitness Planner",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING (Premium SaaS Theme)
# ============================================================================
custom_css = """
<style>
    /* Global Font & Background */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* Main Title Styling */
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Premium Card Styling */
    .premium-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .premium-card:hover {
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.15);
        transform: translateY(-4px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.12);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #999;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Premium Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .rec-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.8rem;
    }
    
    .rec-content {
        font-size: 1rem;
        color: #555;
        line-height: 1.6;
        font-weight: 500;
    }
    
    /* Input Container Styling */
    .input-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9ff 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
    }
    
    /* Color-Coded BMI Categories */
    .bmi-underweight {
        color: #3498db;
        font-weight: 700;
    }
    
    .bmi-normal {
        color: #2ecc71;
        font-weight: 700;
    }
    
    .bmi-overweight {
        color: #f39c12;
        font-weight: 700;
    }
    
    .bmi-obesity {
        color: #e74c3c;
        font-weight: 700;
    }
    
    /* Divider */
    .divider {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
    }
    /* Basic meal plan formatting */
    .meal-plan {
        background: #fdfdfd;
        border: 1px solid #e1e4e8;
        border-radius: 12px;
        padding: 1.2rem;
    }
    .meal-plan .meal-item {
        margin-bottom: 0.8rem;
    }
    .meal-plan .meal-item .meal-name {
        font-weight: 700;
        color: #333;
    }
    .meal-plan .meal-item .meal-desc {
        margin-left: 1rem;
        color: #555;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & ENCODERS
# ============================================================================
@st.cache_resource
def load_model_and_encoders():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'workout_diet_model.pkl')
    encoders_path = os.path.join(base_dir, '..', 'models', 'encoders.pkl')
    
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders

model, encoders = load_model_and_encoders()

# ============================================================================
# CONSTANTS
# ============================================================================
BMI_CATEGORIES = ["Underweight", "Normal weight", "Overweight", "Obesity"]
GOALS = ["muscle_gain", "fat_burn"]
GENDERS = ["Male", "Female"]
# new constant for diet preference
DIET_PREFS = ["Veg", "Non-Veg"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (cm)"""
    return weight / ((height / 100) ** 2)


def get_gender_emoji(gender: str) -> str:
    """Return a simple emoji based on gender string.
    Defaults to neutral user icon if unrecognized.
    """
    if gender.lower() == "male":
        return "👨"
    elif gender.lower() == "female":
        return "👩"
    else:
        return "🧑"  # generic person emoji for other/unknown


def categorize_bmi(bmi):
    """Categorize BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesity"


def get_bmi_color(bmi_category):
    """Get color for BMI category"""
    colors = {
        "Underweight": "#3498db",
        "Normal weight": "#2ecc71",
        "Overweight": "#f39c12",
        "Obesity": "#e74c3c"
    }
    return colors.get(bmi_category, "#95a5a6")


def get_bmi_advice(bmi_category):
    """Get health advice based on BMI category"""
    advice = {
        "Underweight": "💭 Focus on nutritious foods with adequate calories to gain healthy weight.",
        "Normal weight": "✅ Great job! Maintain your current healthy lifestyle.",
        "Overweight": "⚠️ Consider a balanced diet and regular exercise for better health.",
        "Obesity": "🎯 Prioritize gradual lifestyle changes with professional guidance."
    }
    return advice.get(bmi_category, "")


def create_bmi_gauge(bmi, category):
    """Create a beautiful BMI gauge chart using Plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bmi,
        title={'text': "Your BMI", 'font': {'size': 24, 'color': '#333'}},
        delta={'reference': 25, 'increasing': {'color': '#e74c3c'}, 'decreasing': {'color': '#2ecc71'}},
        gauge={
            'axis': {'range': [10, 40], 'tickwidth': 1, 'tickcolor': '#ddd'},
            'bar': {'color': get_bmi_color(category), 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [10, 18.5], 'color': '#e8f4f8'},
                {'range': [18.5, 25], 'color': '#e8f8f0'},
                {'range': [25, 30], 'color': '#fef8e8'},
                {'range': [30, 40], 'color': '#fee8e8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 35
            }
        },
        number={'font': {'size': 36, 'color': get_bmi_color(category), 'family': 'Arial Black'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        font_family="Segoe UI",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig


def create_health_metrics_chart(bmi, gender, goal):
    """Create a professional metrics visualization"""
    categories = ['BMI Status', 'Goal Alignment', 'Plan Quality', 'Adherence Potential']
    
    # Calculate scores
    bmi_score = min((25 - abs(bmi - 25)) / 25 * 100, 100)
    goal_score = 85 if goal == "muscle_gain" else 82
    plan_quality = 95
    adherence = 88
    
    scores = [max(bmi_score, 50), goal_score, plan_quality, adherence]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Your Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color='#999'),
                gridcolor='#ddd'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#333', family='Segoe UI')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        font=dict(family='Segoe UI', size=11, color='#333'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig


def call_gemini_prompt(prompt: str) -> str:
    """Send prompt to Gemini and return text response.

    Uses the modern google.generativeai package with GenerativeModel API.

    Tries multiple sources for the API key:
    1. st.secrets.GOOGLE_API_KEY (from .streamlit/secrets.toml)
    2. os.environ.get("GOOGLE_API_KEY") (from environment variable)
    """

    if genai is None:
        return "[Gemini library not installed]"
    
    # Try to get API key from Streamlit secrets first, then env
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        return "[No API key configured. Please set GOOGLE_API_KEY in .streamlit/secrets.toml or as environment variable]"

    try:
        genai.configure(api_key=api_key)
        # write debug info to sidebar if available
        try:
            st.sidebar.write(f" {api_key[:6]}...")
        except Exception:
            pass
        # candidate model names; some installs use simple names, others
        # require full path.
        # use actual models available in the key's project; listing shows
        # names such as 'models/gemini-pro-latest' or 'models/gemini-3-pro-preview'
        models_to_try = [
            'models/gemini-pro-latest',
            'models/gemini-3-pro-preview',
            'models/gemini-2.5-pro',
            'models/gemini-2.5-flash'
        ]
        last_error = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as err:
                # remember last error for debugging
                last_error = err
                continue
        # none of the models could be instantiated successfully
        debug_msg = ''
        if last_error:
            debug_msg = f" (last error: {last_error})"
        return (
            "[AI service unavailable – no compatible Gemini model found." + debug_msg +
            " Try enabling one of the supported text-generation models in your Google Cloud project]")
    except Exception as e:
        # catch any other unexpected error and return string rather than
        # crashing the Streamlit app
        try:
            st.sidebar.error(f"Gemini error: {e}")
        except Exception:
            pass
        return f"[error calling Gemini: {e}]"


from collections import OrderedDict

def generate_basic_plan(goal: str, diet: str, bmi_cat: str):
    """Return a simple, rule-based 1-day meal plan as fallback.

    Instead of raw text, this now returns an `OrderedDict` mapping meal
    names to descriptions. Rendering logic will detect the dict and
    display it in a nicer formatted way.
    """
    plan = OrderedDict()
    if goal == "muscle_gain":
        plan["Breakfast"] = "Oatmeal with milk, banana, and a scoop of protein powder"
        plan["Mid-morning snack"] = "Greek yogurt with honey"
        plan["Lunch"] = "Grilled chicken wrap with veggies and brown rice"
        plan["Afternoon snack"] = "Handful of nuts and fruit"
        plan["Dinner"] = "Paneer/Tofu stir-fry with quinoa"
        plan["Notes"] = "Calories approx. 2500-2700 depending on portions."
    else:  # fat_burn
        plan["Breakfast"] = "Egg white omelette with spinach"
        plan["Mid-morning snack"] = "Apple slices with peanut butter"
        plan["Lunch"] = "Mixed vegetable salad with chickpeas or grilled fish"
        plan["Afternoon snack"] = "Carrot sticks and hummus"
        plan["Dinner"] = "Steamed veggies with a small portion of lean protein"
        plan["Notes"] = "Keep total calories around 1500-1700 and focus on high fiber."
    return plan


def create_goal_comparison_chart(goal):
    """Create goal effectiveness visualization"""
    goals_data = ['Muscle Gain', 'Fat Burn', 'Strength', 'Endurance', 'Flexibility']
    
    if goal == "muscle_gain":
        values = [95, 60, 88, 50, 55]
        colors = ['#2ecc71', '#95a5a6', '#3498db', '#95a5a6', '#95a5a6']
    else:  # fat_burn
        values = [70, 95, 75, 85, 60]
        colors = ['#95a5a6', '#2ecc71', '#95a5a6', '#3498db', '#95a5a6']
    
    fig = go.Figure(data=go.Bar(
        x=values,
        y=goals_data,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#667eea', width=0)
        ),
        text=[f'{v}%' for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Effectiveness: %{x}%<extra></extra>'
    ))
    
    fig.update_layout(
        font=dict(family='Segoe UI', size=11, color='#333'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#eee', range=[0, 100]),
        yaxis=dict(showgrid=False),
        margin=dict(l=120, r=50, t=20, b=50),
        height=350,
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# HEADER
st.markdown('<h1 class="main-title">💪 FitGenius</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Personalized Fitness & Nutrition Planner</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="font-size: 1rem; color: #999; margin-top: -1rem;">Designed specifically for students • Powered by Machine Learning</p>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - USER INPUT
# ============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">🎯 Your Details</p>', unsafe_allow_html=True)
    
    st.markdown("### Personal Information")
    gender = st.selectbox(
        "👥 Select Your Gender",
        GENDERS,
        key="gender",
        help="Choose your biological gender for personalized recommendations"
    )
    
    height = st.slider(
        "📏 Height (cm)",
        min_value=140.0,
        max_value=220.0,
        value=170.0,
        step=1.0,
        help="Your height in centimeters"
    )
    
    weight = st.slider(
        "⚖️ Weight (kg)",
        min_value=35.0,
        max_value=150.0,
        value=70.0,
        step=0.5,
        help="Your current weight in kilograms"
    )
    
    st.markdown("### Fitness Goal")
    goal = st.selectbox(
        "🎯 Choose Your Goal",
        GOALS,
        key="goal",
        format_func=lambda x: "💪 Muscle Gain" if x == "muscle_gain" else "🔥 Fat Burn",
        help="Select your primary fitness objective"
    )
    
    diet = st.selectbox(
        "🥗 Diet Preference",
        DIET_PREFS,
        key="diet"
    )
    
    st.markdown("---")
    
    # Buttons
    col_btn1, col_btn2 = st.columns([2, 2])
    with col_btn1:
        submit = st.button(
            "🚀 GENERATE MY PLAN",
            use_container_width=True,
            key="generate_btn"
        )
    with col_btn2:
        generate_ai = st.button(
            "🤖 Generate Meal Plan",
            use_container_width=True,
            key="ai_meal_btn"
        )
    
    st.markdown("---")
    st.info("📌 **Tip**: Update your height and weight for accurate recommendations. Consistency is key!")


# ============================================================================
# MAIN CONTENT - RESULTS
# ============================================================================

# compute BMI & category once regardless of button
bmi = calculate_bmi(weight, height)
bmi_cat = categorize_bmi(bmi)

if submit:
    # Prepare data for model prediction
    input_df = pd.DataFrame(
        [[gender, bmi_cat, goal.lower()]],
        columns=['gender', 'bmi_category', 'goal']
    )
    
    # Encode inputs and get predictions
    try:
        for col in ['gender', 'bmi_category', 'goal']:
            input_df[col] = encoders[col].transform(input_df[col])
        
        pred = model.predict(input_df)[0]
        schedule_idx, meal_idx = pred
        
        # Decode predictions
        schedule = encoders['exercise_schedule'].inverse_transform([schedule_idx])[0]
        meal = encoders['meal_plan'].inverse_transform([meal_idx])[0]
        
        # SECTION 1: PROFILE METRICS
        st.markdown('<p class="section-header">📊 Your Health Profile</p>', unsafe_allow_html=True)
        
        metric_cols = st.columns(4)
        
        # Gender Card
        with metric_cols[0]:
            emoji = get_gender_emoji(gender)
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Gender</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{emoji}</div>
                    <div style="color: #667eea; font-weight: 600;">{gender}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        # Height Card
        with metric_cols[1]:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Height</div>
                    <div class="metric-value">{height:.0f}</div>
                    <div style="color: #667eea; font-weight: 600;">cm</div>
                </div>
            ''', unsafe_allow_html=True)
        
        # Weight Card
        with metric_cols[2]:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Weight</div>
                    <div class="metric-value">{weight:.1f}</div>
                    <div style="color: #667eea; font-weight: 600;">kg</div>
                </div>
            ''', unsafe_allow_html=True)
        
        # BMI Card with Color
        bmi_class = f"bmi-{bmi_cat.lower().replace(' ', '-')}"
        with metric_cols[3]:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">BMI</div>
                    <div class="metric-value {bmi_class}">{bmi:.1f}</div>
                    <div style="color: #667eea; font-weight: 600;">{bmi_cat}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'<p style="text-align: center; color: #666; margin-top: 1rem; font-size: 1.05rem; font-weight: 500;">{get_bmi_advice(bmi_cat)}</p>', unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        # SECTION 2: BMI VISUALIZATION
        st.markdown('<p class="section-header">📈 BMI Analysis</p>', unsafe_allow_html=True)
        
        col_gauge, col_metrics = st.columns([1.2, 1])
        
        with col_gauge:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_gauge = create_bmi_gauge(bmi, bmi_cat)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_metrics:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_health = create_health_metrics_chart(bmi, gender, goal)
            st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        # SECTION 3: AI RECOMMENDATIONS
        st.markdown('<p class="section-header">🤖 AI Personalized Recommendations</p>', unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown(f'''
                <div class="recommendation-card fade-in">
                    <div class="rec-title">💪 Workout Schedule</div>
                    <div class="rec-content">{schedule}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown(f'''
                <div class="recommendation-card fade-in">
                    <div class="rec-title">🍽️ Nutrition Plan</div>
                    <div class="rec-content">{meal}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        # SECTION 4: DETAILED ANALYTICS
        st.markdown('<p class="section-header">📊 Detailed Analytics</p>', unsafe_allow_html=True)
        
        col_goal, col_comp = st.columns(2)
        
        with col_goal:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_goal = create_goal_comparison_chart(goal)
            st.plotly_chart(fig_goal, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_comp:
            # Statistics Card
            st.markdown(f'''
                <div class="premium-card" style="padding: 2rem;">
                    <h3 style="color: #333; margin-bottom: 1.5rem;">📋 Plan Summary</h3>
                    <div style="display: flex; flex-direction: column; gap: 1.2rem;">
                        <div style="padding: 1rem; background: #f8f9ff; border-radius: 10px; border-left: 4px solid #667eea;">
                            <p style="margin: 0; color: #999; font-size: 0.9rem; font-weight: 600;">GOAL</p>
                            <p style="margin: 0.3rem 0 0 0; color: #333; font-size: 1.2rem; font-weight: 700;">{goal.replace('_', ' ').title()}</p>
                        </div>
                        <div style="padding: 1rem; background: #f8f9ff; border-radius: 10px; border-left: 4px solid #2ecc71;">
                            <p style="margin: 0; color: #999; font-size: 0.9rem; font-weight: 600;">CATEGORY</p>
                            <p style="margin: 0.3rem 0 0 0; color: #333; font-size: 1.2rem; font-weight: 700;">{bmi_cat}</p>
                        </div>
                        <div style="padding: 1rem; background: #f8f9ff; border-radius: 10px; border-left: 4px solid #f39c12;">
                            <p style="margin: 0; color: #999; font-size: 0.9rem; font-weight: 600;">PLAN PRECISION</p>
                            <p style="margin: 0.3rem 0 0 0; color: #333; font-size: 1.2rem; font-weight: 700;">95%</p>
                        </div>
                        <div style="padding: 1rem; background: #f8f9ff; border-radius: 10px; border-left: 4px solid #3498db;">
                            <p style="margin: 0; color: #999; font-size: 0.9rem; font-weight: 600;">RESULTS TIMELINE</p>
                            <p style="margin: 0.3rem 0 0 0; color: #333; font-size: 1.2rem; font-weight: 700;">4-6 Weeks</p>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        # generate AI meal plan if requested
        if generate_ai:
            # ensure we only call once per session
            key = f"ai_plan_{diet}_{bmi:.1f}_{bmi_cat}_{goal}"
            if key not in st.session_state:
                # Determine calorie target based on goal and BMI category
                if goal == "muscle_gain":
                    calorie_target = "2400-2700"
                    macro_focus = "High protein (40%), moderate carbs (40%), moderate fats (20%)"
                else:  # fat_burn
                    calorie_target = "1500-1800"
                    macro_focus = "High protein (35%), moderate carbs (45%), low fats (20%)"
                
                veg_note = "" if diet == "Veg" else ""
                
                prompt = f"""Create a detailed, realistic 1-day meal plan with the following specifications:

User Profile:
- Goal: {goal.replace('_', ' ').title()}
- BMI Category: {bmi_cat}
- Diet Type: {diet}
- Target Calories: {calorie_target} kcal/day
- Macro Mix: {macro_focus}

Requirements:
1. Create meals that are STUDENT-FRIENDLY and BUDGET-CONSCIOUS
2. Use mostly INDIAN foods and ingredients
3. Include breakfast, lunch, dinner, and 2 snacks
4. Specify calories for each meal
5. Highlight protein sources and amounts
6. Make it realistic and achievable for daily life
7. Format each meal clearly with timing

Format your response as:
**Meal Name** | Time | Calories | Main Items | Protein (g)

Example format:
**Breakfast** | 7:30 AM | 450 kcal | Masala Dosa with chutney, banana | 12g protein

Provide a complete 1-day plan following this format. Ensure total calories match {calorie_target} range."""
                with st.spinner("Generating AI meal plan..."):
                    response = call_gemini_prompt(prompt)
                # debug: show raw API response in sidebar
                try:
                    st.sidebar.write(f"🔍 Gemini returned: {response[:200]}")
                except Exception:
                    pass
                # if the response looks like an error placeholder (starts with
                # '[') or otherwise doesn't seem like a normal plan, substitute
                # a simple default so the user always gets something useful.
                if response.startswith("["):
                    response = generate_basic_plan(goal, diet, bmi_cat)
                    st.session_state[key + "_info"] = (
                        "(Using offline basic meal plan because AI service failed)"
                    )
                st.session_state[key] = response
            ai_plan = st.session_state[key]
            info = st.session_state.get(key + "_info")
            st.markdown('<div class="premium-card" style="padding:1.5rem;height:auto;">', unsafe_allow_html=True)
            st.markdown(f'<h3 class="section-header">🍱 AI Meal Plan</h3>', unsafe_allow_html=True)
            if info:
                st.warning(info)
            # Format response nicely
            if isinstance(ai_plan, dict):
                html = ['<div class="meal-plan">']
                for name, desc in ai_plan.items():
                    html.append(f'<div class="meal-item"><span class="meal-name">{name}</span>: <span class="meal-desc">{desc}</span></div>')
                html.append('</div>')
                st.markdown(''.join(html), unsafe_allow_html=True)
            else:
                # API response - render as readable formatted text
                st.markdown(f'<div style="background:#f8f9ff; padding:1.2rem; border-radius:8px; line-height:1.8;"><pre style="white-space: pre-wrap; font-family:Segoe UI, sans-serif; font-size:0.95rem; color:#333;">{ai_plan}</pre></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # FOOTER SECTION
        st.markdown('''
            <div class="premium-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center;">
                <h3 style="color: white; margin-top: 0;">💡 Get Started Today</h3>
                <p style="color: rgba(255,255,255,0.9); font-size: 1.05rem; margin: 1rem 0;">
                    Follow your personalized plan consistently for 4-6 weeks to see remarkable results. 
                    Remember: Small consistent steps lead to big transformations! 🚀
                </p>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;">Last updated: Today | Next check-in: In 2 weeks</p>
            </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")

# handle AI meal plan generation independent of submit
key = f"ai_plan_{diet}_{bmi:.1f}_{bmi_cat}_{goal}"
if generate_ai or key in st.session_state:
    # if button pressed, possibly generate or refresh; otherwise just display
    if generate_ai and key not in st.session_state:
        # Determine calorie target based on goal and BMI category
        if goal == "muscle_gain":
            calorie_target = "2400-2700"
            macro_focus = "High protein (40%), moderate carbs (40%), moderate fats (20%)"
        else:  # fat_burn
            calorie_target = "1500-1800"
            macro_focus = "High protein (35%), moderate carbs (45%), low fats (20%)"
        
        prompt = f"""Create a detailed, realistic 1-day meal plan with the following specifications:

User Profile:
- Goal: {goal.replace('_', ' ').title()}
- BMI Category: {bmi_cat}
- Diet Type: {diet}
- Target Calories: {calorie_target} kcal/day
- Macro Mix: {macro_focus}

Requirements:
1. Create meals that are STUDENT-FRIENDLY and BUDGET-CONSCIOUS
2. Use mostly INDIAN foods and ingredients
3. Include breakfast, lunch, dinner, and 2 snacks
4. Specify calories for each meal
5. Highlight protein sources and amounts
6. Make it realistic and achievable for daily life
7. Format each meal clearly with timing

Format your response as:
**Meal Name** | Time | Calories | Main Items | Protein (g)

Example format:
**Breakfast** | 7:30 AM | 450 kcal | Masala Dosa with chutney, banana | 12g protein

Provide a complete 1-day plan following this format. Ensure total calories match {calorie_target} range."""
        with st.spinner("Generating AI meal plan..."):
            response = call_gemini_prompt(prompt)
        if response.startswith("["):
            response = generate_basic_plan(goal, diet, bmi_cat)
            st.session_state[key + "_info"] = (
                "ℹ️ Using offline basic meal plan because AI service failed. Set GOOGLE_API_KEY environment variable for personalized plans."
            )
        st.session_state[key] = response

    ai_plan = st.session_state[key]
    info = st.session_state.get(key + "_info")
    st.markdown('<div class="premium-card" style="padding:1.5rem;height:auto;">', unsafe_allow_html=True)
    st.markdown(f'<h3 class="section-header">🍱 AI Meal Plan</h3>', unsafe_allow_html=True)
    if info:
        st.warning(info)
    if isinstance(ai_plan, dict):
        html = ['<div class="meal-plan">']
        for name, desc in ai_plan.items():
            html.append(f'<div class="meal-item"><span class="meal-name">{name}</span>: <span class="meal-desc">{desc}</span></div>')
        html.append('</div>')
        st.markdown(''.join(html), unsafe_allow_html=True)
    else:
        # API response - render as readable formatted text
        st.markdown(f'<div style="background:#f8f9ff; padding:1.2rem; border-radius:8px; line-height:1.8;"><pre style="white-space: pre-wrap; font-family:Segoe UI, sans-serif; font-size:0.95rem; color:#333;">{ai_plan}</pre></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome Section (When not submitted)
    st.markdown('''
        <div class="premium-card" style="text-align: center; padding: 3rem;">
            <h2 style="color: #667eea; margin-bottom: 1rem;">🎉 Welcome to FitGenius</h2>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
                Get your personalized AI Workout Analysis and meal plan powered by AI, 
                designed specifically for students and their busy lifestyles.
            </p>
            <div style="background: #f8f9ff; border-radius: 12px; padding: 2rem; margin: 2rem 0;">
                <h3 style="color: #333; margin-top: 0;">How It Works</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-top: 1.5rem;">
    <div>
        <p style="font-size: 2rem; margin: 0;">1️⃣</p>
        <p style="font-weight: 600; color: #333;">Enter Your Details</p>
        <p style="color: #666; font-size: 0.95rem;">Share your height, weight, and fitness goals</p>
    </div>
    <div>
        <p style="font-size: 2rem; margin: 0;">2️⃣</p>
        <p style="font-weight: 600; color: #333;">AI Analysis</p>
        <p style="color: #666; font-size: 0.95rem;">Our model analyzes your data in real-time</p>
    </div>
    <div>
        <p style="font-size: 2rem; margin: 0;">3️⃣</p>
        <p style="font-weight: 600; color: #333;">Get Recommendations</p>
        <p style="color: #666; font-size: 0.95rem;">Receive personalized workout analysis & meal plans</p>
    </div>
</div>
            </div>
            <p style="color: #999; font-size: 0.9rem; margin-top: 2rem;">👈 Use the sidebar to enter your details and generate your personalized plan!</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Feature Highlights
    st.markdown('<p class="section-header" style="margin-top: 3rem;">✨ Key Features</p>', unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    features = [
        ("🎯", "Personalized Plans", "AI-generated routines tailored to your goals"),
        ("📊", "Health Analytics", "Track BMI, metrics, and progress trends"),
        ("💪", "Science-Backed", "Data-driven recommendations from ML models"),
        ("📱", "Student-Friendly", "Designed for busy schedules")
    ]
    
    for col, (icon, title, desc) in zip([feat_col1, feat_col2, feat_col3, feat_col4], features):
        with col:
            st.markdown(f'''
                <div class="metric-card" style="border-left-width: 0; text-align: center;">
                    <p style="font-size: 2.5rem; margin: 0 0 0.5rem 0;">{icon}</p>
                    <p style="font-weight: 700; color: #333; margin: 0.5rem 0;">{title}</p>
                    <p style="color: #666; font-size: 0.9rem; margin: 0;">{desc}</p>
                </div>
            ''', unsafe_allow_html=True)

# Footer
st.markdown('---')
st.markdown('''
    <p style="text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem;">
    © 2026 FitGenius • Powered by Machine Learning | Built for Students
    </p>
''', unsafe_allow_html=True)
