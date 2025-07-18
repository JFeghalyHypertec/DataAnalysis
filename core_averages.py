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
    st.subheader("🌡️ Average Temperature Table by Core (6 per row + transposed)")

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
            cmap = cm.coolwarm

            fig, (ax1, cax1, ax2, cax2) = plt.subplots(
                nrows=4,
                figsize=(13, 5 + len(df_table) * 0.6),
                gridspec_kw={"height_ratios": [1, 1, 0.05]}
            )
            ax1.axis("off")
            ax2.axis("off")

            # Table 1: standard (6-per-row)
            tbl1 = ax1.table(cellText=df_table.values, loc="center", cellLoc="center")
            for i, row in enumerate(val_rows):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)
                    tbl1[i, j].set_facecolor(color)
            tbl1.auto_set_font_size(False)
            tbl1.set_fontsize(10)
            tbl1.scale(1, 1.5)
            
            # Colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax1, orientation='horizontal')
            cbar.set_label("Temperature (°C)")
            
            # Table 2: transposed (6-per-column)
            # Split into 6 columns, column-wise format
            n_cols = 6
            n_rows = int(np.ceil(len(CORE_LIST) / n_cols))
            transposed_vals = []
            transposed_texts = []

            for col in range(n_cols):
                sub_vals = values[col * n_rows:(col + 1) * n_rows]
                sub_texts = texts[col * n_rows:(col + 1) * n_rows]
                while len(sub_vals) < n_rows:
                    sub_vals.append(np.nan)
                    sub_texts.append("")
                transposed_vals.append(sub_vals)
                transposed_texts.append(sub_texts)

            transposed_vals = np.array(transposed_vals).T.tolist()
            transposed_texts = np.array(transposed_texts).T.tolist()

            tbl2 = ax2.table(cellText=transposed_texts, loc="center", cellLoc="center")
            for i, row in enumerate(transposed_vals):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)
                    tbl2[i, j].set_facecolor(color)
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(10)
            tbl2.scale(1, 1.5)

            # Colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax2, orientation='horizontal')
            cbar.set_label("Temperature (°C)")

            plt.tight_layout()
            st.pyplot(fig)

            # Save as image
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label=f"💾 Save Table as PNG ({uploaded_file.name})",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name}_core_avg_table_with_transpose.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❗ Error processing {uploaded_file.name}: {e}")
            st.exception(e)