import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def calculate_averages_over_time(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with:
    - 'Time': rounded time in minutes
    - 'Average': average temperature across cores per minute (excluding 0s and NaNs)
    """
    # Replace 0 with NaN to skip them in the average
    cleaned_data = core_df.replace(0, np.nan)

    # Create a new column for minute-level buckets
    cleaned_data['minute'] = (cleaned_data.index // 60).astype(int) * 60

    # Compute the average per time row, excluding NaNs
    per_second_avg = cleaned_data.drop(columns='minute').mean(axis=1, skipna=True)

    # Combine into a DataFrame
    temp_df = pd.DataFrame({
        'Time': per_second_avg.index,
        'Average': per_second_avg.values,
        'Minute': cleaned_data['minute']
    })

    # Now group by minute and compute the average of the averages per minute
    result = temp_df.groupby('Minute')['Average'].mean().reset_index()
    result.rename(columns={'Minute': 'Time'}, inplace=True)
    result['Time'] = result['Time'] / 3600  # Optional: convert seconds to minutes

    return result

def plot_percentage_difference(averages: pd.DataFrame, file_name: str) -> plt.Figure:
    """
    Plots the percentage difference of average temperatures over time compared to the first minute.
    """
    baseline = averages['Average'].iloc[0]
    averages['% Difference'] = ((averages['Average'] - baseline) / baseline) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(averages['Time'], averages['% Difference'], marker='o', linestyle='-', color='purple')
    ax.set_title(f"% Difference Over Time for {file_name}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Percentage Difference (%)")
    ax.grid(True)
    
    # Add a horizontal reference line at 0%
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    return fig


def run_core_percentage_difference():
    st.header("Percentage Difference Tool")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=['csv', 'xls', 'xlsx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, header=None)
                else:
                    df = pd.read_excel(uploaded_file, header=None)

                core_df = extract_core_data(df)
                averages = calculate_averages_over_time(core_df)
                fig = plot_percentage_difference(averages, file_name)

                st.subheader(f"📊 Percentage Difference for: `{file_name}`")
                st.pyplot(fig)

                # Save button with dynamic filename
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label=f"💾 Download Percentage Difference for {file_name}",
                    data=buf.getvalue(),
                    file_name=f"{file_name.replace('.','_')}_percentage_difference.png",
                    mime="image/png"
                )
                st.markdown("---")  # Divider between plots

            except Exception as e:
                st.error(f"❗ Error processing file `{file_name}`: {e}")