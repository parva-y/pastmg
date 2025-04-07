import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Campaign Dates
campaign_dates = {
    "derma": ["2025-03-18", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-25", "2025-03-28", "2025-04-01", "2025-04-02", "2025-04-05", "2025-04-07", "2025-04-08", "2025-04-10", "2025-04-11"],
    "diabetic": ["2025-03-06", "2025-03-07", "2025-03-11", "2025-03-13", "2025-03-15", "2025-03-19", "2025-03-23", "2025-03-25", "2025-03-29", "2025-04-01", "2025-04-03"],
    "cardiac": ["2025-03-18", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-25", "2025-04-01", "2025-04-02", "2025-04-05", "2025-04-07", "2025-04-10", "2025-04-12", "2025-04-15", "2025-04-17"],
    "resp": ["2025-03-05", "2025-03-08", "2025-03-12", "2025-03-15", "2025-03-17", "2025-03-19", "2025-03-23", "2025-03-27", "2025-03-30", "2025-04-02", "2025-04-04"]
}

st.title("📊 Test vs Control Performance Analyzer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])

    cohort = st.selectbox("Select Cohort", sorted(df["cohort"].unique()))
    recency = st.selectbox("Select Recency", sorted(df["Recency"].unique()))

    df_filtered = df[(df["cohort"] == cohort) & (df["Recency"] == recency)]

    cohort_dates = pd.to_datetime(campaign_dates[cohort])
    test_start = cohort_dates.min()
    test_end = cohort_dates.max()
    test_length = (test_end - test_start).days + 1

    pre_test_end = test_start - timedelta(days=1)
    pre_test_start = pre_test_end - timedelta(days=test_length - 1)

    st.write(f"🟨 Test Period: {test_start.date()} to {test_end.date()} ({test_length} days)")
    st.write(f"⬜ Pre-Test Period: {pre_test_start.date()} to {pre_test_end.date()}")

    df_filtered["period"] = np.where(
        (df_filtered["date"] >= test_start) & (df_filtered["date"] <= test_end), "Test",
        np.where((df_filtered["date"] >= pre_test_start) & (df_filtered["date"] <= pre_test_end), "Pre-Test", "Other")
    )

    df_period = df_filtered[df_filtered["period"].isin(["Test", "Pre-Test"])]

    # Calculate conversion rates
    df_period["atc_rate"] = df_period["atc"] / df_period["app_opens"]
    df_period["transactor_rate"] = df_period["transactors"] / df_period["app_opens"]
    df_period["order_rate"] = df_period["orders"] / df_period["app_opens"]
    df_period["gmv_per_transactor"] = df_period["gmv"] / df_period["transactors"]

    metrics = ["atc_rate", "transactor_rate", "order_rate", "gmv_per_transactor"]

    st.subheader("📈 Conversion Rates Summary")
    summary = df_period.groupby(["period", "data_set"])[metrics].mean().reset_index()
    st.dataframe(summary.round(4))

    st.subheader("📊 Bar Charts: Conversion Rates")
    for metric in metrics:
        fig, ax = plt.subplots()
        sns.barplot(data=summary, x="data_set", y=metric, hue="period", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        st.pyplot(fig)

    st.subheader("🧪 T-Test Results (Test vs Pre-Test)")
    results = []
    for metric in metrics:
        for group in ["Control Set", "Test Set"]:
            a = df_period[(df_period["data_set"] == group) & (df_period["period"] == "Pre-Test")][metric].dropna()
            b = df_period[(df_period["data_set"] == group) & (df_period["period"] == "Test")][metric].dropna()
            if len(a) > 1 and len(b) > 1:
                t_stat, p_val = ttest_ind(b, a, equal_var=False)
                results.append({
                    "Group": group,
                    "Metric": metric,
                    "T-Statistic": t_stat,
                    "P-Value": p_val
                })

    result_df = pd.DataFrame(results)
    result_df["Significant (<0.05)"] = result_df["P-Value"] < 0.05
    st.dataframe(result_df.round(4))

    st.caption("T-test compares pre vs test period for each metric within each group.")
