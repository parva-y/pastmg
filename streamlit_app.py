import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

st.set_page_config(layout="wide")
st.title("📊 Test vs Control Performance Analyzer")

# Campaign Dates
campaign_dates = {
    "derma": ["2025-03-18", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-25", "2025-03-28", "2025-04-01", "2025-04-02", "2025-04-05", "2025-04-07", "2025-04-08", "2025-04-10", "2025-04-11"],
    "diabetic": ["2025-03-06", "2025-03-07", "2025-03-11", "2025-03-13", "2025-03-15", "2025-03-19", "2025-03-23", "2025-03-25", "2025-03-29", "2025-04-01", "2025-04-03"],
    "cardiac": ["2025-03-18", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-25", "2025-04-01", "2025-04-02", "2025-04-05", "2025-04-07", "2025-04-10", "2025-04-12", "2025-04-15", "2025-04-17"],
    "resp": ["2025-03-05", "2025-03-08", "2025-03-12", "2025-03-15", "2025-03-17", "2025-03-19", "2025-03-23", "2025-03-27", "2025-03-30", "2025-04-02", "2025-04-04"]
}

# Keep uploaded file fixed
if "data" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        st.session_state["data"] = df
else:
    df = st.session_state["data"]

if "data" in st.session_state:
    all_cohorts = sorted(df["cohort"].dropna().unique().tolist())
    all_recencies = sorted(df["Recency"].dropna().unique().tolist())

    cohort = st.selectbox("Select Cohort", ["All"] + all_cohorts)
    recency = st.selectbox("Select Recency", ["All"] + all_recencies)

    # Filter based on user selection
    df_filtered = df.copy()
    if cohort != "All":
        df_filtered = df_filtered[df_filtered["cohort"] == cohort]
    if recency != "All":
        df_filtered = df_filtered[df_filtered["Recency"] == recency]

    # Define campaign window based on cohort
    if cohort != "All":
        cohort_dates = pd.to_datetime(campaign_dates[cohort])
        test_start = cohort_dates.min()
        test_end = cohort_dates.max()
    else:
        all_dates = [pd.to_datetime(date) for dates in campaign_dates.values() for date in dates]
        test_start = min(all_dates)
        test_end = max(all_dates)

    test_length = (test_end - test_start).days + 1
    pre_test_end = test_start - timedelta(days=1)
    pre_test_start = pre_test_end - timedelta(days=test_length - 1)

    df_filtered["period"] = np.where(
        (df_filtered["date"] >= test_start) & (df_filtered["date"] <= test_end), "Test",
        np.where((df_filtered["date"] >= pre_test_start) & (df_filtered["date"] <= pre_test_end), "Pre-Test", "Other")
    )

    df_period = df_filtered[df_filtered["period"].isin(["Test", "Pre-Test"])]

    # Calculate conversion rates based on audience size
    df_period["atc_rate"] = df_period["atc"] / df_period["audience_size"]
    df_period["transactor_rate"] = df_period["transactors"] / df_period["audience_size"]
    df_period["order_rate"] = df_period["orders"] / df_period["audience_size"]
    df_period["gmv_per_transactor"] = df_period["gmv"] / df_period["transactors"]

    metrics = ["atc_rate", "transactor_rate", "order_rate", "gmv_per_transactor"]

    st.markdown(f"🟨 Test Period: `{test_start.date()}` to `{test_end.date()}` ({test_length} days)")
    st.markdown(f"⬜ Pre-Test Period: `{pre_test_start.date()}` to `{pre_test_end.date()}`")

    st.subheader("📈 Conversion Rates Summary")
    summary = df_period.groupby(["period", "data_set"])[metrics].mean().reset_index()
    st.dataframe(summary.round(4), use_container_width=True)
    
    # Bar Charts (side-by-side)
    st.subheader("📊 Bar Charts: Conversion Rates")
    bar_cols = st.columns(2)
    for idx, metric in enumerate(metrics):
        with bar_cols[idx % 2]:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            sns.barplot(data=summary, x="data_set", y=metric, hue="period", ax=ax_bar)
            ax_bar.set_title(metric.replace("_", " ").title())
            plt.tight_layout()
            st.pyplot(fig_bar)


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
    st.dataframe(result_df.round(4), use_container_width=True)

    st.caption("T-test compares pre vs test period for each metric within each group.")
