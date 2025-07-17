import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import numpy as np

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]  # All 26 cores

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    if time.iloc[0] > 1e6:
        time = time / 1000
    time = time - time.iloc[0]
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def run_display_core_avg_table():
    st.subheader("🌡️ Average Temperature Table by Core (6 per row, heatmap-style)")

    uploaded_files = st.file_uploader(
        "📂 Upload one or more OCCT CSV Files",
        type=["csv"],
        key="avg-table-upload",
        accept_multiple_files=True
    )
    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 {uploaded_file.name}")
        try:
            df = pd.read_csv(uploaded_file, header=None)
            core_df = extract_core_data(df)
            avg_temps = core_df.mean()

            # Format core text and store values for color mapping
            values = [avg_temps.get(core) for core in CORE_LIST]
            texts = [
                f"{core} = {val:.2f}°C" if pd.notnull(val) else f"{core} = N/A"
                for core, val in zip(CORE_LIST, values)
            ]

            # Group into rows of 6
            text_rows = [texts[i:i+6] for i in range(0, len(texts), 6)]
            val_rows = [values[i:i+6] for i in range(0, len(values), 6)]

            # Pad last row if needed
            while len(text_rows[-1]) < 6:
                text_rows[-1].append("")
                val_rows[-1].append(np.nan)

            df_table = pd.DataFrame(text_rows)

            # Normalize values for coloring
            flat_vals = [v for row in val_rows for v in row if not pd.isna(v)]
            vmin, vmax = min(flat_vals), max(flat_vals)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.coolwarm  # or use plt.cm.viridis, etc.

            fig, ax = plt.subplots(figsize=(12, 2 + len(df_table)*0.5))
            ax.axis('off')
            tbl = ax.table(cellText=df_table.values, loc='center', cellLoc='center')

            # Color each cell based on temperature value
            for i, row in enumerate(val_rows):
                for j, val in enumerate(row):
                    if pd.notnull(val):
                        color = cmap(norm(val))
                    else:
                        color = (1, 1, 1, 1)  # white
                    cell = tbl[i, j]  # FIXED: use i, not i+1
                    cell.set_facecolor(color)

            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            plt.tight_layout()

            st.pyplot(fig)

            # Save as image
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label=f"💾 Save Table as PNG ({uploaded_file.name})",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name}_core_avg_table.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")