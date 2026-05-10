import streamlit as st

st.set_page_config(
    page_title="Diabetes Analysis",
    page_icon="🩺",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Header */
        .header {
            background: linear-gradient(90deg, #1a6b3c 0%, #28a745 100%);
            padding: 1.5rem 2rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .header h1 {
            color: white;
            font-size: 2.4rem;
            margin: 0;
            letter-spacing: 1px;
        }
        .header p {
            color: #d4edda;
            font-size: 1rem;
            margin: 0.3rem 0 0;
        }

        /* Footer */
        .footer {
            background: #1a1a2e;
            color: #adb5bd;
            padding: 1rem 2rem;
            border-radius: 10px;
            margin-top: 2rem;
            text-align: center;
            font-size: 0.85rem;
        }
        .footer a { color: #28a745; text-decoration: none; }

        /* Section cards */
        .card {
            background: #f8f9fa;
            border-left: 5px solid #28a745;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header">
        <h1>🩺 Diabetes Analysis</h1>
        <p>Team 8 · Python Architects · Python Hackathon May 2026</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Data Exploration", "Predictions", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Use the menu above to navigate between sections.")

# ── BODY ─────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.subheader("Project Overview")
    st.markdown(
        """
        <div class="card">
            This dashboard provides an end-to-end analysis of the <b>Pima Indians Diabetes Dataset</b>.
            Use the sidebar to explore the data, run predictions, or learn more about the project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "768")
    col2.metric("Features", "8")
    col3.metric("Diabetic Cases", "268 (34.9%)")

    st.divider()
    st.subheader("Key Insights")
    st.markdown(
        """
        - **Glucose** is the strongest predictor of diabetes.
        - Patients with **BMI > 30** are at significantly higher risk.
        - **Age** and **family history** (Diabetes Pedigree Function) are important secondary factors.
        """
    )

elif page == "Data Exploration":
    st.subheader("Data Exploration")
    st.info("Upload your dataset or connect to the cleaned data to explore distributions, correlations, and outliers.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        import pandas as pd
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)
        st.bar_chart(df.select_dtypes("number").mean())
    else:
        st.markdown(
            '<div class="card">No file uploaded yet. Please upload a CSV file to begin exploration.</div>',
            unsafe_allow_html=True,
        )

elif page == "Predictions":
    st.subheader("Diabetes Risk Prediction")
    st.markdown("Enter patient details below to estimate diabetes risk.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        pregnancies   = c1.number_input("Pregnancies",          0, 20,  1)
        glucose       = c2.number_input("Glucose (mg/dL)",      0, 300, 120)
        blood_pressure= c1.number_input("Blood Pressure (mmHg)",0, 200, 70)
        skin_thickness= c2.number_input("Skin Thickness (mm)",  0, 100, 20)
        insulin       = c1.number_input("Insulin (IU/mL)",      0, 900, 80)
        bmi           = c2.number_input("BMI",                  0.0, 70.0, 25.0, step=0.1)
        dpf           = c1.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age           = c2.number_input("Age",                  1, 120, 30)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Placeholder heuristic — swap with a real model call
        risk_score = (glucose / 200 + bmi / 70 + dpf) / 3
        if risk_score > 0.5:
            st.error(f"High risk of diabetes (score: {risk_score:.2f}). Please consult a physician.")
        else:
            st.success(f"Low risk of diabetes (score: {risk_score:.2f}). Keep up a healthy lifestyle!")

elif page == "About":
    st.subheader("About This Project")
    st.markdown(
        """
        <div class="card">
            <b>Team 8 – Python Architects</b><br>
            Python Hackathon · May 2026<br><br>
            This project uses machine learning and data visualisation techniques to analyse
            diabetes risk factors in the Pima Indians dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Tech Stack**
        - Python · Pandas · Scikit-learn
        - Streamlit · Matplotlib · Seaborn
        """
    )

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        © 2026 Team 8 · Python Architects &nbsp;|&nbsp;
        Python Hackathon May 2026 &nbsp;|&nbsp;
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True,
)
