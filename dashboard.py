"""
Team 08 — Python Architects | HUPA-UC Diabetes Dataset Dashboard
Team 08 — Python Architects | HUPA-UC Diabetes Dataset

Run with: streamlit run dashboard.py
Dependencies: pip install streamlit plotly pandas
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="T1DM Patient Dashboard",
    page_icon="🩺",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE = r"C:\Data Analyst_Resources\Python_Hackathon\HUPA-UC Diabetes Dataset"

@st.cache_data
def load_data():
    df = pd.read_csv(rf"{BASE}\Practice_files\Cleaned_Diabetes_Data.csv")
    demo = pd.read_csv(rf"{BASE}\T1DM_patient_sleep_demographics_with_race.csv")
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    data = df.merge(demo, on="Patient_ID", how="left")
    return df, demo, data

df, demo, data = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 T1DM Patient Monitoring Dashboard")
st.markdown(
    "**HUPA-UC Diabetes Dataset** | Type 1 Diabetes Mellitus — Monitoring & Predictive Analysis"
)
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
patients = ["All Patients"] + sorted(data["Patient_ID"].unique().tolist())
selected_patient = st.sidebar.selectbox("Select Patient", patients)

filtered = data if selected_patient == "All Patients" else data[data["Patient_ID"] == selected_patient]

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Blood Glucose",   f"{filtered['Blood_Glucose_mg_dl'].mean():.1f} mg/dL")
k2.metric("Hypoglycemia Events", f"{(filtered['Blood_Glucose_mg_dl'] < 70).mean()*100:.1f}%")
k3.metric("Hyperglycemia Events",f"{(filtered['Blood_Glucose_mg_dl'] > 180).mean()*100:.1f}%")
k4.metric("Avg Heart Rate",      f"{filtered['Heart_Rate_bpm'].mean():.1f} bpm")
k5.metric("Avg Steps / interval",f"{filtered['Step_count'].mean():.1f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demographics & Sleeps",
    "👤 Descriptive Analysis",
    "🏃 Insulin & Activity",
    "🤖 ML Model Results",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Glucose Trends
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Blood Glucose Over Time")

    if selected_patient == "All Patients":
        daily = (
            filtered
            .groupby(["Date", "Patient_ID"])["Blood_Glucose_mg_dl"]
            .mean()
            .reset_index()
        )
        fig_line = px.line(
            daily, x="Date", y="Blood_Glucose_mg_dl", color="Patient_ID",
            title="Daily Average Blood Glucose by Patient",
            labels={"Blood_Glucose_mg_dl": "Avg Glucose (mg/dL)"},
        )
    else:
        fig_line = px.line(
            filtered, x="TimeStamp", y="Blood_Glucose_mg_dl",
            title=f"Blood Glucose Over Time — {selected_patient}",
            labels={"Blood_Glucose_mg_dl": "Glucose (mg/dL)", "TimeStamp": "Time"},
        )

    fig_line.add_hline(y=70,  line_dash="dash", line_color="red",    annotation_text="Hypo < 70")
    fig_line.add_hline(y=180, line_dash="dash", line_color="orange",  annotation_text="Hyper > 180")
    fig_line.add_hline(y=100, line_dash="dot",  line_color="green",   annotation_text="Normal 100")
    st.plotly_chart(fig_line, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        hourly_gluc = filtered.groupby("Hour")["Blood_Glucose_mg_dl"].mean().reset_index()
        fig_hourly = px.bar(
            hourly_gluc, x="Hour", y="Blood_Glucose_mg_dl",
            title="Avg Blood Glucose by Hour of Day",
            labels={"Blood_Glucose_mg_dl": "Avg Glucose (mg/dL)"},
            color="Blood_Glucose_mg_dl", color_continuous_scale="RdYlGn_r",
        )
        fig_hourly.add_hline(y=70,  line_dash="dash", line_color="red")
        fig_hourly.add_hline(y=180, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_hourly, use_container_width=True)

    with c2:
        fig_hist = px.histogram(
            filtered, x="Blood_Glucose_mg_dl", nbins=50,
            title="Blood Glucose Distribution",
            labels={"Blood_Glucose_mg_dl": "Blood Glucose (mg/dL)"},
            color_discrete_sequence=["steelblue"],
        )
        fig_hist.add_vline(x=70,  line_dash="dash", line_color="red",    annotation_text="Hypo")
        fig_hist.add_vline(x=180, line_dash="dash", line_color="orange",  annotation_text="Hyper")
        st.plotly_chart(fig_hist, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Demographics & Sleep
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Patient Demographics & Sleep Quality")

    c1, c2 = st.columns(2)

    with c1:
        gender_df = demo["Gender"].value_counts().reset_index()
        gender_df.columns = ["Gender", "Count"]
        st.plotly_chart(
            px.pie(gender_df, names="Gender", values="Count",
                   title="Gender Distribution", hole=0.4),
            use_container_width=True,
        )

    with c2:
        race_df = demo["Race"].value_counts().reset_index()
        race_df.columns = ["Race", "Count"]
        st.plotly_chart(
            px.bar(race_df, x="Race", y="Count",
                   title="Race / Ethnicity Distribution",
                   color="Count", color_continuous_scale="Blues"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)

    with c3:
        st.plotly_chart(
            px.histogram(demo, x="Age", nbins=15,
                         title="Age Distribution of Patients",
                         color_discrete_sequence=["coral"]),
            use_container_width=True,
        )

    with c4:
        st.plotly_chart(
            px.scatter(
                demo,
                x="Average Sleep Duration (hrs)", y="Sleep Quality (1-10)",
                color="Race", size="% with Sleep Disturbances",
                hover_data=["Patient_ID", "Age", "Gender"],
                title="Sleep Duration vs Sleep Quality",
            ),
            use_container_width=True,
        )

    race_sleep = demo.groupby("Race")["% with Sleep Disturbances"].mean().reset_index()
    st.plotly_chart(
        px.bar(race_sleep, x="Race", y="% with Sleep Disturbances",
               title="Avg Sleep Disturbances by Race (%)",
               color="% with Sleep Disturbances", color_continuous_scale="Reds"),
        use_container_width=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Insulin & Activity
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Insulin Usage & Physical Activity Patterns")

    c1, c2 = st.columns(2)

    with c1:
        basal_hourly = filtered.groupby("Hour")["Basal_Insulin_Rate_Unit_hr"].mean().reset_index()
        st.plotly_chart(
            px.bar(basal_hourly, x="Hour", y="Basal_Insulin_Rate_Unit_hr",
                   title="Avg Basal Insulin Rate by Hour",
                   labels={"Basal_Insulin_Rate_Unit_hr": "Basal Rate (U/hr)"},
                   color_discrete_sequence=["mediumpurple"]),
            use_container_width=True,
        )

    with c2:
        steps_hourly = filtered.groupby("Hour")["Step_count"].mean().reset_index()
        st.plotly_chart(
            px.bar(steps_hourly, x="Hour", y="Step_count",
                   title="Avg Step Count by Hour of Day",
                   labels={"Step_count": "Avg Steps"},
                   color_discrete_sequence=["mediumseagreen"]),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)

    with c3:
        carb_hourly = filtered.groupby("Hour")["Carbohydrate_Intake_Grams"].mean().reset_index()
        st.plotly_chart(
            px.bar(carb_hourly, x="Hour", y="Carbohydrate_Intake_Grams",
                   title="Avg Carbohydrate Intake by Hour",
                   labels={"Carbohydrate_Intake_Grams": "Avg Carbs (g)"},
                   color_discrete_sequence=["sandybrown"]),
            use_container_width=True,
        )

    with c4:
        sample = filtered.sample(min(5000, len(filtered)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="Step_count", y="Blood_Glucose_mg_dl",
            color="Heart_Rate_bpm",
            title="Blood Glucose vs Physical Activity",
            labels={
                "Step_count": "Step Count",
                "Blood_Glucose_mg_dl": "Blood Glucose (mg/dL)",
                "Heart_Rate_bpm": "Heart Rate (bpm)",
            },
            color_continuous_scale="Viridis",
            opacity=0.5,
        )
        fig_scatter.add_hline(y=70,  line_dash="dash", line_color="red")
        fig_scatter.add_hline(y=180, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Bolus insulin events
    bolus_events = (
        filtered[filtered["Bolus_Insulin_Dose_Units"] > 0]
        .groupby("Hour")["Bolus_Insulin_Dose_Units"]
        .mean()
        .reset_index()
    )
    if not bolus_events.empty:
        st.plotly_chart(
            px.bar(bolus_events, x="Hour", y="Bolus_Insulin_Dose_Units",
                   title="Avg Bolus Insulin Dose by Hour (meal doses only)",
                   labels={"Bolus_Insulin_Dose_Units": "Avg Bolus Dose (U)"},
                   color_discrete_sequence=["tomato"]),
            use_container_width=True,
        )

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — ML Model Results
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Hypoglycemia Risk Prediction — ML Model Comparison")
    st.markdown(
        "Three models were trained to predict **hypoglycemia risk** (glucose < 70 mg/dL) "
        "using patient physiological, insulin, and activity features."
    )

    model_results = pd.DataFrame({
        "Model":          ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy (%)":   [93.41, 93.48, 93.66],
        "Precision (%)":  [0.00,  51.62, 62.12],
        "Recall (%)":     [0.00,  15.24,  2.01],
        "F1 Score (%)":   [0.00,  23.53,  3.90],
        "ROC AUC (%)":    [60.21, 77.12, 75.05],
        "Best Use":       [
            "Interpretability / explainability",
            "✅ Best overall (highest ROC AUC)",
            "Precision-focused / fewest false alarms",
        ],
    })

    st.dataframe(
        model_results.style.highlight_max(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"],
            color="#d4edda",
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_roc = px.bar(
            model_results, x="Model", y="ROC AUC (%)",
            title="ROC AUC Score by Model",
            color="ROC AUC (%)", color_continuous_scale="Greens",
            text="ROC AUC (%)",
        )
        fig_roc.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_roc.update_layout(yaxis_range=[50, 85])
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        categories = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"]
        colors = ["royalblue", "seagreen", "tomato"]
        fig_radar = go.Figure()
        for i, row in model_results.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories],
                theta=categories,
                fill="toself",
                name=row["Model"],
                line_color=colors[i],
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Model Performance Radar Chart",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Key Findings")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info(
            "**Best Model: Random Forest**  \n"
            "Highest ROC AUC (77.12%) — best at distinguishing hypoglycemia from normal readings."
        )
    with f2:
        st.warning(
            "**Class Imbalance Challenge**  \n"
            "Only 6.6% of readings are hypoglycemic, making recall difficult across all models."
        )
    with f3:
        st.success(
            "**XGBoost Precision**  \n"
            "Highest precision (62.12%) — fewest false alarms when predicting hypoglycemia risk."
        )

    st.divider()

    # ── Hyperglycemia section ────────────────────────────────────────────────
    st.subheader("Hyperglycemia Risk Prediction — ML Model Comparison")
    st.markdown(
        "Three models were trained to predict **hyperglycemia risk** (glucose > 180 mg/dL). "
        "Blood glucose was included as a feature, making this a near-perfect classification task."
    )

    hyper_results = pd.DataFrame({
        "Model":          ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy (%)":   [99.99, 100.00, 99.90],
        "Precision (%)":  [99.99, 100.00, 99.69],
        "Recall (%)":     [99.96,  99.99, 99.86],
        "F1 Score (%)":   [99.97,  99.99, 99.77],
        "ROC AUC (%)":    [100.0,  100.0, 100.0],
        "Best Use":       [
            "✅ Best overall (highest ROC AUC, tied)",
            "Perfect precision — zero false positives",
            "Advanced diabetes monitoring",
        ],
    })

    st.dataframe(
        hyper_results.style.highlight_max(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"],
            color="#d4edda",
        ),
        use_container_width=True,
        hide_index=True,
    )

    hc1, hc2 = st.columns(2)

    with hc1:
        fig_hyper_bar = px.bar(
            hyper_results, x="Model", y="ROC AUC (%)",
            title="Hyperglycemia — ROC AUC Score by Model",
            color="Accuracy (%)", color_continuous_scale="Blues",
            text="Accuracy (%)",
        )
        fig_hyper_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_hyper_bar.update_layout(yaxis_range=[99, 100.5])
        st.plotly_chart(fig_hyper_bar, use_container_width=True)

    with hc2:
        categories = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"]
        colors = ["royalblue", "seagreen", "tomato"]
        fig_hyper_radar = go.Figure()
        for i, row in hyper_results.iterrows():
            fig_hyper_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories],
                theta=categories,
                fill="toself",
                name=row["Model"],
                line_color=colors[i],
                opacity=0.6,
            ))
        fig_hyper_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[99, 100])),
            title="Hyperglycemia — Model Performance Radar",
        )
        st.plotly_chart(fig_hyper_radar, use_container_width=True)

    hf1, hf2, hf3 = st.columns(3)
    with hf1:
        st.info(
            "**All Models Near-Perfect**  \n"
            "ROC AUC ~100% across all three models for hyperglycemia detection."
        )
    with hf2:
        st.success(
            "**Random Forest: Zero False Positives**  \n"
            "100% precision — never incorrectly flags a normal reading as hyperglycemia."
        )
    with hf3:
        st.warning(
            "**Why So High?**  \n"
            "Blood glucose is included as a feature — it directly determines the label (> 180 mg/dL)."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center'><small>"
    "Team 08 — Python Architects &nbsp;|&nbsp; HUPA-UC Diabetes Dataset &nbsp;|&nbsp; Hackathon 2024"
    "</small></div>",
    unsafe_allow_html=True,
)
