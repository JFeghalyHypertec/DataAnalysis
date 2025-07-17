import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import BytesIO
import numpy as np

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]  # All 26 cores

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    time = time - time.iloc[0]
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def run_display_core_avg_table():
    st.subheader("🌡️ Average Temperature Table by Core")

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

            values = [avg_temps.get(core) for core in CORE_LIST]
            texts = [
                f"{core} = {val:.2f}°C" if pd.notnull(val) else f"{core} = N/A"
                for core, val in zip(CORE_LIST, values)
            ]

            # Group into rows of 6 for original heatmap-style table
            text_rows = [texts[i:i+6] for i in range(0, len(texts), 6)]
            val_rows = [values[i:i+6] for i in range(0, len(values), 6)]

            while len(text_rows[-1]) < 6:
                text_rows[-1].append("")
                val_rows[-1].append(np.nan)

            df_table = pd.DataFrame(text_rows)

            # Build the 6-core-per-column vertical layout
            col1_cores = CORE_LIST[:13]
            col2_cores = CORE_LIST[13:]
            col1_vals = values[:13]
            col2_vals = values[13:]

            max_len = max(len(col1_cores), len(col2_cores))
            col1_cores += [""] * (max_len - len(col1_cores))
            col2_cores += [""] * (max_len - len(col2_cores))
            col1_vals += [np.nan] * (max_len - len(col1_vals))
            col2_vals += [np.nan] * (max_len - len(col2_vals))

            col1_text = [
                f"{c} = {v:.2f}°C" if c and pd.notnull(v) else ""
                for c, v in zip(col1_cores, col1_vals)
            ]
            col2_text = [
                f"{c} = {v:.2f}°C" if c and pd.notnull(v) else ""
                for c, v in zip(col2_cores, col2_vals)
            ]
            vertical_table = pd.DataFrame({
                "Group A": col1_text,
                "Group B": col2_text
            })
            vertical_vals = list(zip(col1_vals, col2_vals))

            # Normalize for both tables
            flat_vals = [v for v in values if pd.notnull(v)]
            vmin, vmax = min(flat_vals), max(flat_vals)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.coolwarm

            # Plot both tables side by side
            fig, (ax1, ax2, cax) = plt.subplots(
                ncols=3,
                figsize=(16, 2 + max(len(df_table), len(vertical_table)) * 0.5),
                gridspec_kw={"width_ratios": [6, 6, 0.3]}
            )

            ax1.axis("off")
            ax2.axis("off")

            tbl1 = ax1.table(cellText=df_table.values, loc="center", cellLoc="center")
            tbl2 = ax2.table(cellText=vertical_table.values, loc="center", cellLoc="center")

            # Heatmap for first table
            for i, row in enumerate(val_rows):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)
                    tbl1[i, j].set_facecolor(color)

            # Heatmap for second vertical table
            for i, (v1, v2) in enumerate(vertical_vals):
                color1 = cmap(norm(v1)) if pd.notnull(v1) else (1, 1, 1, 1)
                color2 = cmap(norm(v2)) if pd.notnull(v2) else (1, 1, 1, 1)
                tbl2[i, 0].set_facecolor(color1)
                tbl2[i, 1].set_facecolor(color2)

            for tbl in [tbl1, tbl2]:
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                tbl.scale(1, 1.5)

            # Colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Temperature (°C)")

            plt.tight_layout()
            st.pyplot(fig)

            # Save as image
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label=f"💾 Save Combined Table as PNG ({uploaded_file.name})",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name}_combined_avg_tables.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❗ Error processing {uploaded_file.name}: {e}")