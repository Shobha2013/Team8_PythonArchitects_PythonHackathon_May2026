import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Diabetes Lifestyle & Health Dashboard",
    layout="wide"
)
csv_folder = r'C:\Users\ruchi\OneDrive\Documents\Python Hackathon May 2026\HUPA-UC Diabetes Dataset-20250820T010637Z-1-001\HUPA-UC Diabetes Dataset'
cleaned_combined_data_file = r"C:\Temporary\combined_df_data.csv"
output_file2 = r"C:\Temporary\dashboard_data.csv"
demographic_file = os.path.join(csv_folder, "T1DM_patient_sleep_demographics_with_race.csv")


demographic_df = pd.read_csv(demographic_file)
dataset_df = pd.read_csv(cleaned_combined_data_file)
combined_df = pd.merge(
    demographic_df,
    dataset_df,
    on='Patient_ID',
    how='left'
)

st.title("Diabetes Lifestyle & Health Dashboard")
st.markdown(
    "A dashboard view of blood glucose control, sleep quality, and sleep disturbance "
    "patterns across demographic groups."
)

avg_glucose = combined_df['Blood_Glucose_mg_dl'].mean()
max_glucose = combined_df['Blood_Glucose_mg_dl'].max()
hyper = (combined_df['Blood_Glucose_mg_dl'] > 300).mean() * 100

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Average Blood Glucose", f"{avg_glucose:.1f} mg/dL")
kpi_col2.metric("Maximum Glucose Level", f"{max_glucose:.0f} mg/dL")
kpi_col3.metric("Hyperglycemia Percentage", f"{hyper:.1f}%")

st.markdown("---")
st.markdown("## Charts")
chart_col1, chart_col2, chart_col3 = st.columns(3)

fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.regplot(
    x='Average Sleep Duration (hrs)',
    y='Heart_Rate_bpm',
    data=combined_df,
    ax=ax1
)
ax1.set_title('Sleep Duration vs Heart Rate')
chart_col1.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(5, 4))
hist_plot = sns.histplot(
    combined_df['Sleep Quality (1-10)'],
    kde=True,
    bins=5,
    ax=ax2
)
ax2.set_title('Distribution of Sleep Quality')
for bar in ax2.patches:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(
            f'{int(height)}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom'
        )
chart_col2.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(5, 4))
bar_plot = sns.barplot(
    x='Race',
    y='% with Sleep Disturbances',
    data=combined_df,
    palette='Set2',
    ax=ax3
)
ax3.set_title('Sleep Disturbances by Race')
for bar in ax3.patches:
    height = bar.get_height()
    ax3.annotate(
        f'{height:.1f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords='offset points',
        ha='center',
        va='bottom'
    )
ax3.tick_params(axis='x', rotation=30)
chart_col3.pyplot(fig3)


st.markdown("---")
st.markdown("## Correlation Charts")
chart_col1, chart_col2, chart_col3 = st.columns(3)

# Combined Health Metrics Correlation Heatmap
fig4, ax4 = plt.subplots(figsize=(6, 5))
sns.heatmap(
    combined_df[
        [
            'Blood_Glucose_mg_dl',
            'Heart_Rate_bpm',
            'Step_count',
            'Calories_burned',
            'Average Sleep Duration (hrs)',
            'Sleep Quality (1-10)',
            '% with Sleep Disturbances',
            'Age'
        ]
    ].corr(),
    annot=True,
    cmap='coolwarm',
    center=0,
    ax=ax4,
    cbar_kws={'shrink': 0.7}
)
ax4.set_title('Combined Health Metrics Correlation')
chart_col1.pyplot(fig4)
chart_col1.caption("Figure 1: Correlation between health metrics including glucose, heart rate, activity, sleep, and age.")

# Insulin Management & Glucose Control Correlation
fig_insulin, ax_insulin = plt.subplots(figsize=(6, 5))
biomarker_cols_insulin = [
    'Blood_Glucose_mg_dl',
    'Basal_Insulin_Rate_Unit_hr',
    'Bolus_Insulin_Dose_Units',
    'Carbohydrate_Intake_Grams',
    'Age'
]
correlation_matrix_insulin = combined_df[biomarker_cols_insulin].corr()
sns.heatmap(
    correlation_matrix_insulin,
    annot=True,
    fmt='.2f',
    cmap='RdYlBu_r',
    center=0,
    ax=ax_insulin,
    cbar_kws={'shrink': 0.7}
)
ax_insulin.set_title('Insulin Management & Glucose Control')
chart_col2.pyplot(fig_insulin)
chart_col2.caption("Figure 2: Relationships between insulin dosing, carbohydrate intake, age, and blood glucose.")

# Physical Activity & Sleep Metrics Correlation
fig_activity, ax_activity = plt.subplots(figsize=(6, 5))
biomarker_cols_activity = [
    'Step_count',
    'Calories_burned',
    'Heart_Rate_bpm',
    'Average Sleep Duration (hrs)',
    'Sleep Quality (1-10)',
    '% with Sleep Disturbances'
]
correlation_matrix_activity = combined_df[biomarker_cols_activity].corr()
sns.heatmap(
    correlation_matrix_activity,
    annot=True,
    fmt='.2f',
    cmap='RdYlBu_r',
    center=0,
    ax=ax_activity,
    cbar_kws={'shrink': 0.7}
)
ax_activity.set_title('Physical Activity & Sleep Quality')
chart_col3.pyplot(fig_activity)
chart_col3.caption("Figure 3: Associations between activity, heart rate, and sleep metrics.")

st.markdown("---")
st.markdown("## Dataset Distribution")
chart_col4, chart_col5, chart_col6 = st.columns(3)

fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
category_counts = combined_df['Race'].value_counts()
ax_pie.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 9}
)
ax_pie.axis('equal')
ax_pie.set_title('Patient Distribution by Race')
chart_col4.pyplot(fig_pie)
chart_col4.caption("Figure: Distribution of patients by race from the dataset, showing demographic composition as a pie chart.")
