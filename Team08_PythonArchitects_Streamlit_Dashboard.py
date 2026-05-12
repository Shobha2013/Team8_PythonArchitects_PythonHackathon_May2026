"""
Team 08 — Python Architects | HUPA-UC Diabetes Dataset Dashboard
Covers: Data Cleaning → Descriptive → Prescriptive → ML Models
Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes data Dashboard — Team 08- Python Architects",
    page_icon="🩺",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE = r"C:\Data Analyst_Resources\Python_Hackathon\HUPA-UC Diabetes Dataset"

@st.cache_data
def load_data():
    df   = pd.read_csv(rf"{BASE}\Practice_files\Cleaned_Diabetes_Data.csv")
    demo = pd.read_csv(rf"{BASE}\T1DM_patient_sleep_demographics_with_race.csv")
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    data = df.merge(demo, on="Patient_ID", how="left")
    return df, demo, data

df, demo, data = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Patient Monitoring Dashboard")
st.markdown(
    "**Team 08 — Python Architects** &nbsp;|&nbsp; HUPA-UC Diabetes Dataset &nbsp;|&nbsp; "
    "Type 1 Diabetes Mellitus — Monitoring & Predictive Analysis"
)
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
patients = ["All Patients"] + sorted(data["Patient_ID"].unique().tolist())
selected_patient = st.sidebar.selectbox("Select Patient", patients)
filtered = data if selected_patient == "All Patients" else data[data["Patient_ID"] == selected_patient]

st.sidebar.divider()
st.sidebar.markdown("**Team 08 — Python Architects**")
st.sidebar.markdown("HUPA-UC Diabetes Dataset")
st.sidebar.markdown("Hackathon 2026")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Patients",      f"{data['Patient_ID'].nunique()}")
k2.metric("Total Records",       f"{len(df):,}")
k3.metric("Avg Blood Glucose",   f"{filtered['Blood_Glucose_mg_dl'].mean():.2f} mg/dL")
k4.metric("Hypoglycemia %",      f"{(filtered['Blood_Glucose_mg_dl'] < 70).mean()*100:.2f}%")
k5.metric("Hyperglycemia %",     f"{(filtered['Blood_Glucose_mg_dl'] > 180).mean()*100:.2f}%")
k6.metric("Avg Calories Burned", f"{filtered['Calories_burned'].mean():.2f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Demographics & Data Cleaning",
    "📊 Descriptive Analysis",
    "💡 Prescriptive Analysis",
    "🤖 Predictive Analysis(ML)"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMOGRAPHICS & Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("👥 Patient Demographics anlysis")

    c1, c2 = st.columns(2)
    with c1:
        gdf = demo["Gender"].value_counts().reset_index()
        gdf.columns = ["Gender", "Count"]
        st.plotly_chart(
            px.pie(gdf, names="Gender", values="Count",
                   title="Gender Distribution", hole=0.4,
                   color_discrete_sequence=["#3498db", "#e91e8c"]),
            use_container_width=True,
        )
    with c2:
        rdf = demo["Race"].value_counts().reset_index()
        rdf.columns = ["Race", "Count"]
        st.plotly_chart(
            px.bar(rdf.sort_values("Count"), x="Count", y="Race",
                   orientation="h",
                   title="Race / Ethnicity Distribution",
                   color="Count", color_continuous_scale="Blues"),
            use_container_width=True,
        )

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            px.histogram(demo, x="Age", nbins=15,
                         title="Age Distribution of Patients",
                         color_discrete_sequence=["coral"],
                         labels={"Age": "Patient Age (years)"}),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            px.scatter(
                demo,
                x="Average Sleep Duration (hrs)", y="Sleep Quality (1-10)",
                color="Race",
                size="% with Sleep Disturbances",
                hover_data=["Patient_ID", "Age", "Gender"],
                title="Sleep Duration vs Sleep Quality by Race",
            ),
            use_container_width=True,
        )

    st.divider()

    race_sleep = demo.groupby("Race")["% with Sleep Disturbances"].mean().reset_index()
    fig_race_sleep = px.bar(
        race_sleep.sort_values("% with Sleep Disturbances"),
        x="Race", y="% with Sleep Disturbances",
        title="Average Sleep Disturbances by Race (%)",
        color="% with Sleep Disturbances",
        color_continuous_scale="Reds",
        text="% with Sleep Disturbances",
    )
    fig_race_sleep.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_race_sleep, use_container_width=True)

    st.divider()

    st.markdown("#### Blood Glucose Variability by Age Group")
    dc4 = data.copy()
    dc4["Age_Group"] = pd.cut(
        dc4["Age"],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=["0–20", "21–30", "31–40", "41–50", "51–60", "60+"],
    )
    age_var = dc4.groupby("Age_Group", observed=True)["Blood_Glucose_mg_dl"].agg(
        ["mean", "std"]
    ).reset_index()
    age_var.columns = ["Age Group", "Avg Glucose", "Std Dev"]

    c5, c6 = st.columns(2)
    with c5:
        fig_age_avg = px.bar(
            age_var, x="Age Group", y="Avg Glucose",
            title="Average Blood Glucose by Age Group",
            color="Avg Glucose", color_continuous_scale="RdYlGn_r",
            text="Avg Glucose",
        )
        fig_age_avg.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_age_avg, use_container_width=True)
    with c6:
        fig_age_std = px.bar(
            age_var, x="Age Group", y="Std Dev",
            title="Glucose Variability (Std Dev) by Age Group",
            color="Std Dev", color_continuous_scale="Oranges",
            text="Std Dev",
        )
        fig_age_std.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_age_std, use_container_width=True)
    st.info(
        "**Insight:** The 51–60 age group exhibits the highest glucose variability, "
        "suggesting significantly less stable blood sugar control — this demographic "
        "may require closer monitoring and personalized management strategies."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Descriptive Analysis — Key Questions")

    st.markdown("#### Glucose Zone Distribution & High-Risk Patients")
    c1, c2 = st.columns(2)

    with c1:
        bins_z = [-float("inf"), 70, 180, float("inf")]
        lbl_z  = ["Hypoglycemia (<70)", "Normal (70–180)", "Hyperglycemia (>180)"]
        fc     = filtered.copy()
        fc["Glucose_Zone"] = pd.cut(fc["Blood_Glucose_mg_dl"], bins=bins_z, labels=lbl_z)
        zone_cnt = fc["Glucose_Zone"].value_counts().reindex(lbl_z)

        fig_zone = px.pie(
            values=zone_cnt.values,
            names=zone_cnt.index,
            title="Glucose Zone Distribution",
            color=zone_cnt.index,
            color_discrete_map={
                "Hypoglycemia (<70)":   "#d62728",
                "Normal (70–180)":      "#2ca02c",
                "Hyperglycemia (>180)": "#1f77b4",
            },
            hole=0.35,
        )
        st.plotly_chart(fig_zone, use_container_width=True)
        st.info(
            "**Insight:** Most readings fall in the **Normal zone (71.5%)**. "
            "However, **21.7% Hyperglycemia** and **6.8% Hypoglycemia** indicate "
            "glucose swings that need closer monitoring."
        )

    with c2:
        patient_avg = df.groupby("Patient_ID")["Blood_Glucose_mg_dl"].mean().reset_index()
        high_risk   = patient_avg[patient_avg["Blood_Glucose_mg_dl"] > 180].sort_values(
            "Blood_Glucose_mg_dl"
        )
        fig_hr = px.bar(
            high_risk,
            x="Blood_Glucose_mg_dl", y="Patient_ID",
            orientation="h",
            title="High-Risk Patients (Avg Glucose > 180 mg/dL)",
            labels={"Blood_Glucose_mg_dl": "Avg Glucose (mg/dL)", "Patient_ID": "Patient"},
            color="Blood_Glucose_mg_dl",
            color_continuous_scale="Reds",
            text="Blood_Glucose_mg_dl",
        )
        fig_hr.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_hr.add_vline(x=180, line_dash="dash", line_color="orange",
                         annotation_text="Threshold 180")
        st.plotly_chart(fig_hr, use_container_width=True)
        st.warning(
            "**Insight:** Patients such as **HUPA0017P** and **HUPA0020P** show persistent "
            "hyperglycemia (~200 mg/dL), indicating poor long-term glucose control and "
            "requiring immediate clinical attention."
        )

    st.divider()

    st.markdown("#### Hourly Blood Glucose Patterns")
    c3, c4 = st.columns(2)

    with c3:
        hourly_avg = filtered.groupby("Hour")["Blood_Glucose_mg_dl"].mean().reset_index()
        peak_hour  = int(hourly_avg.loc[hourly_avg["Blood_Glucose_mg_dl"].idxmax(), "Hour"])
        peak_val   = hourly_avg["Blood_Glucose_mg_dl"].max()
        fig_h_avg  = px.line(
            hourly_avg, x="Hour", y="Blood_Glucose_mg_dl",
            title="Average Blood Glucose by Hour of Day",
            labels={"Blood_Glucose_mg_dl": "Avg Glucose (mg/dL)"},
            markers=True,
        )
        fig_h_avg.add_hline(y=180, line_dash="dash", line_color="orange",
                            annotation_text="Hyper > 180")
        fig_h_avg.add_hline(y=70, line_dash="dash", line_color="red",
                            annotation_text="Hypo < 70")
        fig_h_avg.add_vline(x=peak_hour, line_dash="dot", line_color="purple",
                            annotation_text=f"Peak Hour {peak_hour}")
        st.plotly_chart(fig_h_avg, use_container_width=True)
        st.info(
            f"**Insight:** Glucose peaks at **Hour {peak_hour} ({peak_val:.2f} mg/dL)**, "
            "suggesting dinner intake or reduced evening activity contributes to glucose spikes."
        )

    with c4:
        hourly_std = filtered.groupby("Hour")["Blood_Glucose_mg_dl"].std().reset_index()
        hourly_std.columns = ["Hour", "Glucose_Std"]
        peak_var     = int(hourly_std.loc[hourly_std["Glucose_Std"].idxmax(), "Hour"])
        peak_std_val = hourly_std["Glucose_Std"].max()
        fig_var = px.bar(
            hourly_std, x="Hour", y="Glucose_Std",
            title="Glucose Variability (Std Dev) by Hour",
            labels={"Glucose_Std": "Std Deviation (mg/dL)"},
            color="Glucose_Std",
            color_continuous_scale="Oranges",
        )
        fig_var.add_vline(x=peak_var, line_dash="dot", line_color="red",
                          annotation_text=f"Peak Hour {peak_var}")
        st.plotly_chart(fig_var, use_container_width=True)
        st.info(
            f"**Insight:** Highest variability at **Hour {peak_var} (SD = {peak_std_val:.2f} mg/dL)** — "
            "may be linked to evening meals, insulin effects, or reduced activity late in the day."
        )

    st.divider()

    st.markdown("#### Carbohydrate Intake & Hyperglycemia Occurrence")
    c5, c6 = st.columns(2)

    with c5:
        hourly_carbs = filtered.groupby("Hour")["Carbohydrate_Intake_Grams"].mean().reset_index()
        peak_carb_hr  = int(hourly_carbs.loc[hourly_carbs["Carbohydrate_Intake_Grams"].idxmax(), "Hour"])
        peak_carb_val = hourly_carbs["Carbohydrate_Intake_Grams"].max()
        fig_carbs = px.bar(
            hourly_carbs, x="Hour", y="Carbohydrate_Intake_Grams",
            title="Avg Carbohydrate Intake by Hour of Day",
            labels={"Carbohydrate_Intake_Grams": "Avg Carbs (g)"},
            color="Carbohydrate_Intake_Grams",
            color_continuous_scale="Greens",
        )
        fig_carbs.add_vline(x=peak_carb_hr, line_dash="dot", line_color="darkgreen",
                            annotation_text=f"Peak Hour {peak_carb_hr}")
        st.plotly_chart(fig_carbs, use_container_width=True)
        st.info(
            f"**Insight:** Highest carb intake at **Hour {peak_carb_hr} ({peak_carb_val:.2f} g avg)**, "
            "reflecting dinner or late-evening snacking patterns."
        )

    with c6:
        hyper_df2 = pd.DataFrame({
            "Category": ["Normal (≤180 mg/dL)", "Hyperglycemia (>180 mg/dL)"],
            "Count":    [
                int((filtered["Blood_Glucose_mg_dl"] <= 180).sum()),
                int((filtered["Blood_Glucose_mg_dl"] > 180).sum()),
            ]
        })
        fig_hyper2 = px.bar(
            hyper_df2, x="Category", y="Count",
            title="Hyperglycemia Occurrence (>180 mg/dL)",
            color="Category",
            color_discrete_map={
                "Normal (≤180 mg/dL)":        "#2ecc71",
                "Hyperglycemia (>180 mg/dL)": "#e74c3c",
            },
            text="Count",
        )
        fig_hyper2.update_traces(textposition="outside")
        fig_hyper2.update_layout(showlegend=False)
        st.plotly_chart(fig_hyper2, use_container_width=True)
        st.warning(
            "**Insight:** ~67,123 hyperglycemic readings across the dataset. "
            "Frequent spikes increase long-term complication risk — may indicate "
            "insufficient insulin response, poor meal control, or stress/sleep disturbances."
        )

    st.divider()

    st.markdown("#### Carbohydrates & Insulin vs Blood Glucose")
    c7, c8 = st.columns(2)

    with c7:
        sample_carb = filtered.sample(min(3000, len(filtered)), random_state=42)
        carb_corr   = filtered["Carbohydrate_Intake_Grams"].corr(filtered["Blood_Glucose_mg_dl"])
        fig_carb_sc = px.scatter(
            sample_carb, x="Carbohydrate_Intake_Grams", y="Blood_Glucose_mg_dl",
            title=f"Carbs vs Blood Glucose (r = {carb_corr:.2f})",
            labels={
                "Carbohydrate_Intake_Grams": "Carbohydrate Intake (g)",
                "Blood_Glucose_mg_dl":       "Blood Glucose (mg/dL)",
            },
            opacity=0.4,
            color_discrete_sequence=["steelblue"],
        )
        fig_carb_sc.add_hline(y=180, line_dash="dash", line_color="orange")
        fig_carb_sc.add_hline(y=70,  line_dash="dash", line_color="red")
        st.plotly_chart(fig_carb_sc, use_container_width=True)
        st.info(
            f"**Insight:** Correlation = **{carb_corr:.2f}** — virtually no linear relationship. "
            "Glucose ranges widely even at near-zero carb intake."
        )

    with c8:
        sample_ins = filtered.sample(min(3000, len(filtered)), random_state=42)
        ins_corr   = filtered["Basal_Insulin_Rate_Unit_hr"].corr(filtered["Blood_Glucose_mg_dl"])
        fig_ins    = px.scatter(
            sample_ins, x="Basal_Insulin_Rate_Unit_hr", y="Blood_Glucose_mg_dl",
            title=f"Basal Insulin Rate vs Blood Glucose (r = {ins_corr:.2f})",
            labels={
                "Basal_Insulin_Rate_Unit_hr": "Basal Insulin (U/hr)",
                "Blood_Glucose_mg_dl":         "Blood Glucose (mg/dL)",
            },
            opacity=0.4,
            trendline="ols",
            color_discrete_sequence=["mediumpurple"],
        )
        fig_ins.add_hline(y=180, line_dash="dash", line_color="orange")
        fig_ins.add_hline(y=70,  line_dash="dash", line_color="red")
        st.plotly_chart(fig_ins, use_container_width=True)
        st.info(
            "**Insight:** Weak correlation — insulin dosing is not the sole control mechanism. "
            "Activity, meal timing, and stress also play significant roles."
        )

    st.divider()

    st.markdown("#### Calories Burned & Step Count by Patient")
    c9, c10 = st.columns(2)

    with c9:
        avg_cal     = filtered["Calories_burned"].mean()
        cal_by_hour = filtered.groupby("Hour")["Calories_burned"].mean().reset_index()
        fig_cal = px.bar(
            cal_by_hour, x="Hour", y="Calories_burned",
            title=f"Avg Calories Burned by Hour (Overall Avg = {avg_cal:.2f} cal)",
            labels={"Calories_burned": "Avg Calories Burned"},
            color="Calories_burned",
            color_continuous_scale="Purples",
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        st.info(
            f"**Insight:** Overall avg calories burned = **{avg_cal:.2f} cal/interval**. "
            "Low values reflect high-frequency measurements capturing many resting periods."
        )

    with c10:
        steps_by_pt = df.groupby("Patient_ID")["Step_count"].sum().reset_index()
        steps_by_pt.columns = ["Patient_ID", "Total_Steps"]
        fig_steps = px.bar(
            steps_by_pt.sort_values("Total_Steps"),
            x="Total_Steps", y="Patient_ID",
            orientation="h",
            title="Total Step Count by Patient",
            labels={"Total_Steps": "Total Steps", "Patient_ID": "Patient"},
            color="Total_Steps",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_steps, use_container_width=True)
        st.info(
            "**Insight:** Wide variation in steps across patients. "
            "High step patients may reflect exceptional activity or potential data outliers."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💡 Prescriptive Analysis — Clinical Recommendations")

    # ── Heart Rate Zone vs Blood Glucose (Box Plot) ───────────────────────────
    st.markdown("#### Blood Glucose by Heart Rate Zone")
    dc2 = data.copy()
    dc2["HR_Zone"] = pd.cut(
        dc2["Heart_Rate_bpm"],
        bins=[40, 80, 110, 150],
        labels=["Low (40–80 bpm)", "Moderate (80–110 bpm)", "High (110–150 bpm)"],
    )
    hr_box_sample = dc2.dropna(subset=["HR_Zone"]).sample(min(5000, len(dc2)), random_state=42)
    fig_hr_box = px.box(
        hr_box_sample, x="HR_Zone", y="Blood_Glucose_mg_dl",
        title="Glucose Levels by Heart Rate Zone",
        labels={
            "HR_Zone":             "Heart Rate Zone",
            "Blood_Glucose_mg_dl": "Blood Glucose (mg/dL)",
        },
        color="HR_Zone",
        color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
    )
    fig_hr_box.add_hline(y=180, line_dash="dash", line_color="orange",
                         annotation_text="Hyper > 180")
    fig_hr_box.add_hline(y=70, line_dash="dash", line_color="red",
                         annotation_text="Hypo < 70")
    st.plotly_chart(fig_hr_box, use_container_width=True)
    st.success(
        "**Insight:** **Moderate HR (80–110 bpm)** correlates with improved glucose control — "
        "muscles take up glucose more efficiently at this intensity. "
        "Very high HR can trigger stress hormones that raise glucose instead of lowering it."
    )

    st.divider()

    # ── Step Count vs Blood Glucose (Violin Chart) ────────────────────────────
    st.markdown("#### Distribution of Blood Glucose Across Step Ranges")
    dc3 = data.copy()
    dc3["Step_Bin"] = pd.cut(dc3["Step_count"], bins=10)
    bin_labels    = [str(b) for b in dc3["Step_Bin"].cat.categories]
    violin_colors = ["yellow", "purple"]

    fig_violin = go.Figure()
    for i, label in enumerate(bin_labels):
        bin_data = dc3[dc3["Step_Bin"].astype(str) == label]["Blood_Glucose_mg_dl"].dropna()
        if len(bin_data) == 0:
            continue
        fig_violin.add_trace(go.Violin(
            y=bin_data,
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=violin_colors[i % 2],
            line_color="black",
            opacity=0.7,
            x0=label,
        ))

    fig_violin.update_layout(
        title="Distribution of Blood Glucose Across Step Ranges",
        xaxis_title="Step Count Bins",
        yaxis_title="Blood Glucose (mg/dL)",
        showlegend=False,
        xaxis_tickangle=-45,
        plot_bgcolor="white",
    )
    fig_violin.add_hline(y=180, line_dash="dash", line_color="orange",
                         annotation_text="Hyper > 180")
    fig_violin.add_hline(y=70,  line_dash="dash", line_color="red",
                         annotation_text="Hypo < 70")
    st.plotly_chart(fig_violin, use_container_width=True)
    st.info(
        "**Insight:** Patients with higher step counts tend to show **lower and more stable** "
        "blood glucose levels. Increased physical activity is associated with better glucose control. "
        "Most patients are highly sedentary — increasing daily steps is a key recommendation."
    )

    st.divider()

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.markdown("#### Lifestyle & Health Factors Correlated with Blood Glucose")
    numeric_cols = [
        "Blood_Glucose_mg_dl", "Calories_burned", "Heart_Rate_bpm",
        "Step_count", "Basal_Insulin_Rate_Unit_hr",
        "Bolus_Insulin_Dose_Units", "Carbohydrate_Intake_Grams",
    ]
    corr_matrix = data[numeric_cols].corr().round(2)
    fig_corr = px.imshow(
        corr_matrix,
        title="Correlation Heatmap",
        color_continuous_scale="plasma",
        text_auto=".2f",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.success(
        "**Insight:** Supports personalized diabetes management by showing how daily habits "
        "and insulin administration impact blood sugar control — enabling more targeted "
        "lifestyle and treatment recommendations."
    )

    st.divider()

    # ── Time-of-Day Insulin Adjustment ────────────────────────────────────────
    st.markdown("#### When Should Insulin Be Adjusted Based on Time-of-Day Glucose Patterns?")

    hourly_stats = data.groupby("Hour").agg(
        mean_glucose=("Blood_Glucose_mg_dl", "mean"),
        std_glucose=("Blood_Glucose_mg_dl",  "std"),
        max_glucose=("Blood_Glucose_mg_dl",  "max"),
    ).round(2).reset_index()

    def _risk(row):
        if row["max_glucose"] > 300:
            return "CRITICAL"
        elif row["mean_glucose"] > 200:
            return "HIGH"
        elif row["mean_glucose"] > 150:
            return "MODERATE"
        elif row["mean_glucose"] < 100:
            return "LOW"
        else:
            return "NORMAL"

    hourly_stats["risk_level"] = hourly_stats.apply(_risk, axis=1)

    risk_colors = {
        "CRITICAL": "#e74c3c",
        "HIGH":     "#f39c12",
        "MODERATE": "#f1c40f",
        "LOW":      "#3498db",
        "NORMAL":   "#2ecc71",
    }

    tp1, tp2 = st.columns([2, 1])
    with tp1:
        fig_time = px.bar(
            hourly_stats, x="Hour", y="mean_glucose",
            color="risk_level",
            color_discrete_map=risk_colors,
            error_y="std_glucose",
            title="Hourly Glucose Levels with Insulin Adjustment Risk Assessment",
            labels={
                "mean_glucose": "Mean Glucose (mg/dL)",
                "Hour":         "Hour of Day",
                "risk_level":   "Risk Level",
            },
            text="mean_glucose",
        )
        fig_time.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_time.add_hline(y=180, line_dash="dash", line_color="orange",
                           annotation_text="High (180)")
        fig_time.add_hline(y=150, line_dash="dash", line_color="green",
                           annotation_text="Target (150)")
        fig_time.add_hline(y=100, line_dash="dash", line_color="blue",
                           annotation_text="Low (100)")
        st.plotly_chart(fig_time, use_container_width=True)

    with tp2:
        risk_counts = hourly_stats["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_risk_pie = px.pie(
            risk_counts, names="Risk Level", values="Count",
            title="Distribution of Risk Levels by Hour",
            color="Risk Level",
            color_discrete_map=risk_colors,
        )
        st.plotly_chart(fig_risk_pie, use_container_width=True)

    st.info(
        "**Insight:** Hours with CRITICAL or HIGH risk require a **basal insulin increase of 10–20%**. "
        "NORMAL hours should maintain current dosing. LOW-risk hours suggest reducing basal by 10–20% "
        "to prevent hypoglycemia. Adjustments should be reassessed every 2–3 days."
    )

    st.divider()

    # ── Sleep Quality Distribution ────────────────────────────────────────────
    st.markdown("#### Distribution of Sleep Quality Among Participants")

    demo_sq = demo.copy()
    demo_sq["Sleep_Category"] = demo_sq["Sleep Quality (1-10)"].apply(
        lambda q: "Low" if q <= 3 else ("High" if q > 7 else "Medium")
    )
    sleep_counts = demo_sq["Sleep_Category"].value_counts().reset_index()
    sleep_counts.columns = ["Sleep Quality Category", "Count"]

    sq1, sq2 = st.columns([1, 1])
    with sq1:
        fig_sleep = px.pie(
            sleep_counts,
            names="Sleep Quality Category",
            values="Count",
            title="Sleep Quality Distribution (Low / Medium / High)",
            color="Sleep Quality Category",
            color_discrete_map={
                "Low":    "#e74c3c",
                "Medium": "#f39c12",
                "High":   "#2ecc71",
            },
            hole=0.4,
        )
        fig_sleep.update_traces(texttemplate="%{label}<br>%{value} (%{percent:.2%})")
        st.plotly_chart(fig_sleep, use_container_width=True)

    with sq2:
        fig_sleep_bar = px.bar(
            sleep_counts.sort_values("Sleep Quality Category"),
            x="Sleep Quality Category", y="Count",
            title="Participant Count by Sleep Quality Category",
            color="Sleep Quality Category",
            color_discrete_map={
                "Low":    "#e74c3c",
                "Medium": "#f39c12",
                "High":   "#2ecc71",
            },
            text="Count",
        )
        fig_sleep_bar.update_traces(texttemplate="%{text}", textposition="outside")
        fig_sleep_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_sleep_bar, use_container_width=True)

    st.info(
        "**Insight:** The majority of participants fall in the **Medium sleep quality** category (scores 4–7). "
        "Only a small fraction achieve High quality sleep (>7). Poor sleep is linked to impaired glucose regulation — "
        "improving sleep habits is a key lifestyle recommendation for better diabetes management."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predictive Analysis (ML Models)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── HYPOGLYCEMIA ──────────────────────────────────────────────────────────
    st.subheader("🔵 Hypoglycemia Risk Prediction (Blood Glucose < 70 mg/dL)")
    st.markdown(
        "Three models trained to predict **hypoglycemia risk**. "
        "Blood glucose is **excluded** from features to avoid data leakage. "
        "Class imbalance: only **6.6%** of readings are hypoglycemic."
    )

    hypo_df = pd.DataFrame({
        "Model":         ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy (%)":  [93.41, 93.48, 93.66],
        "Precision (%)": [0.00,  51.62, 62.12],
        "Recall (%)":    [0.00,  15.24,  2.01],
        "F1 Score (%)":  [0.00,  23.53,  3.90],
        "ROC AUC (%)":   [60.21, 77.12, 75.05],
        "Best Use":      [
            "Interpretability / explainability",
            "✅ Best ROC AUC — recommended",
            "Highest precision (fewest false alarms)",
        ],
    })

    st.dataframe(
        hypo_df.style.highlight_max(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"],
            color="#d4edda",
        ).highlight_min(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"],
            color="#f8d7da",
        ),
        use_container_width=True, hide_index=True,
    )

    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"]
    colors  = ["royalblue", "seagreen", "tomato"]

    hc1, hc2 = st.columns(2)
    with hc1:
        fig_hypo_radar = go.Figure()
        for i, row in hypo_df.iterrows():
            fig_hypo_radar.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill="toself",
                name=row["Model"],
                line_color=colors[i],
                opacity=0.6,
            ))
        fig_hypo_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Hypoglycemia — Model Performance Radar",
        )
        st.plotly_chart(fig_hypo_radar, use_container_width=True)

    with hc2:
        fig_hypo_grouped = px.bar(
            hypo_df.melt(id_vars="Model", value_vars=metrics,
                         var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model",
            barmode="group",
            title="Hypoglycemia — All Metrics by Model",
            labels={"Score": "Score (%)"},
            color_discrete_sequence=colors,
            text="Score",
        )
        fig_hypo_grouped.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_hypo_grouped, use_container_width=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.info(
            "**Best Overall: Random Forest**  \n"
            "Highest ROC AUC (77.12%) — best at distinguishing hypoglycemia from normal readings."
        )
    with h2:
        st.warning(
            "**Class Imbalance Challenge**  \n"
            "Only 6.6% of readings are hypoglycemic — very difficult to achieve high recall."
        )
    with h3:
        st.success(
            "**XGBoost — Highest Precision**  \n"
            "62.12% precision — fewest false alarms when predicting hypoglycemia risk."
        )

    st.divider()

    # ── HYPERGLYCEMIA ─────────────────────────────────────────────────────────
    st.subheader("🔴 Hyperglycemia Risk Prediction (Blood Glucose > 180 mg/dL)")
    st.markdown(
        "Three models trained to predict **hyperglycemia risk**. "
        "Blood glucose is **included** as a feature — it directly determines the label (>180 mg/dL). "
        "Class distribution: **21.7%** hyperglycemia."
    )

    hyper_df = pd.DataFrame({
        "Model":         ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy (%)":  [99.99, 100.00, 99.90],
        "Precision (%)": [99.99, 100.00, 99.69],
        "Recall (%)":    [99.96,  99.99, 99.86],
        "F1 Score (%)":  [99.97,  99.99, 99.77],
        "ROC AUC (%)":   [100.0,  100.0, 100.0],
        "Best Use":      [
            "Early warning system — High recall",
            "✅ Zero false positives (100% precision)",
            "Advanced diabetes monitoring",
        ],
    })

    st.dataframe(
        hyper_df.style.highlight_max(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC AUC (%)"],
            color="#d4edda",
        ),
        use_container_width=True, hide_index=True,
    )

    hh1, hh2 = st.columns(2)
    with hh1:
        fig_hyper_bar = px.bar(
            hyper_df, x="Model", y="F1 Score (%)",
            title="Hyperglycemia — F1 Score by Model",
            color="F1 Score (%)", color_continuous_scale="Blues",
            text="F1 Score (%)",
        )
        fig_hyper_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_hyper_bar.update_layout(yaxis_range=[99.50, 100.20])
        st.plotly_chart(fig_hyper_bar, use_container_width=True)

    with hh2:
        fig_hyper_radar = go.Figure()
        for i, row in hyper_df.iterrows():
            fig_hyper_radar.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
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
            "Blood glucose is included as a feature — it directly determines the label (>180 mg/dL)."
        )

    st.divider()

    st.subheader("📊 Hypoglycemia vs Hyperglycemia — ROC AUC Comparison")
    compare_df = pd.DataFrame({
        "Model":         ["Logistic Regression", "Random Forest", "XGBoost"],
        "Hypoglycemia":  [60.21, 77.12, 75.05],
        "Hyperglycemia": [100.0, 100.0, 100.0],
    }).melt(id_vars="Model", var_name="Task", value_name="ROC AUC (%)")

    fig_compare = px.bar(
        compare_df, x="Model", y="ROC AUC (%)", color="Task",
        barmode="group",
        title="ROC AUC: Hypoglycemia vs Hyperglycemia Prediction",
        color_discrete_map={"Hypoglycemia": "steelblue", "Hyperglycemia": "tomato"},
        text="ROC AUC (%)",
    )
    fig_compare.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_compare, use_container_width=True)
    st.markdown(
        "> **Takeaway:** Hyperglycemia prediction is near-perfect because blood glucose itself "
        "is a feature. Hypoglycemia is much harder — the class is rare (6.6%) and blood glucose "
        "was excluded to ensure a fair, leakage-free model."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center'><small>"
    "Team 08 — Python Architects &nbsp;|&nbsp; HUPA-UC Diabetes Dataset &nbsp;|&nbsp; "
    "Hackathon 2025 &nbsp;|&nbsp; "
    "Data Cleaning → Descriptive → Prescriptive → Predictive Analysis"
    "</small></div>",
    unsafe_allow_html=True,
)
