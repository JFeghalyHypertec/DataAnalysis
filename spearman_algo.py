# streamlit_rank_similarity.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import spearmanr
from io import BytesIO
# import your existing extract_core_data function
from coreHeatmapPlot import extract_core_data  

def run_spearman_rank_similarity():
    st.set_page_config(page_title="Spearman Rank Similarity", layout="wide")
    st.title("🔥 Core-Ranking Similarity Across Tests")

    uploaded_files = st.file_uploader(
        "Upload one or more CPU test files (CSV or XLSX)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []  # will hold (filename, Series of averages)

        # 1) Process each file: extract core DataFrame & compute averages
        for up in uploaded_files:
            try:
                # read raw table (no header) so extract_core_data can find CORE_LIST in row 1
                if up.name.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(up, header=None)
                else:
                    df = pd.read_csv(up, header=None)

                core_df = extract_core_data(df)
                averages = core_df[core_df != 0].mean()
                results.append((up.name, averages))

                # 2) Draw the bar chart for this file
                fig, ax = plt.subplots(figsize=(4, 6))
                avg_rev = averages[::-1]  # so Core 0 is at top
                bars = ax.barh(avg_rev.index, avg_rev.values, color="gray")
                ax.set_title(f"Avg Temp per Core\n{up.name}", pad=10)
                ax.set_xlabel("°C")
                ax.set_xlim(avg_rev.min() - 5, avg_rev.max() + 5)

                # annotate
                for bar, val in zip(bars, avg_rev.values):
                    ax.text(val + 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}°C",
                            va="center",
                            ha="left",
                            fontsize=9)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing **{up.name}**: {e}")

        # 3) If more than one test, compute & show Spearman correlation heatmap
        if len(results) > 1:
            st.subheader("🔗 Spearman Rank Correlation Matrix")

            # build DataFrame: rows = tests, cols = cores
            df_avgs = pd.DataFrame({
                fname: avgs
                for fname, avgs in results
            }).T

            # rank each row (hottest → rank 1)
            df_ranks = df_avgs.rank(axis=1, method="average", ascending=False)

            # spearman corr between cores
            corr = df_ranks.corr(method="spearman")

            # plot heatmap
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar_kws={"label": "ρ"},
                ax=ax2
            )
            ax2.set_title("Spearman ρ between cores across Tests", pad=12)
            ax2.set_xticklabels(corr.columns, rotation=90, fontsize=6)
            ax2.set_yticklabels(corr.index, rotation=0, fontsize=6)
            plt.tight_layout()
            st.pyplot(fig2)