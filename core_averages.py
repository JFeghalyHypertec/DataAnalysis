import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import BytesIO
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            # Choose a colormap and normalization for temperature values
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
            cmap = cm.get_cmap("coolwarm")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

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

            # --- Second table: 6 cores per column, stacked below the first table ---
            n_cols = 6
            n_rows = int(np.ceil(len(CORE_LIST) / n_cols))
            col_texts = []
            col_vals = []
            for col in range(n_cols):
                col_cores = CORE_LIST[col * n_rows : (col + 1) * n_rows]
                col_values = values[col * n_rows : (col + 1) * n_rows]
                col_texts.append([
                    f"{c} = {v:.2f}°C" if c and pd.notnull(v) else ""
                    for c, v in zip(col_cores, col_values)
                ])
                col_vals.append(col_values)
            # Transpose for table display
            second_table_text = list(map(list, zip(*col_texts)))
            second_table_vals = list(map(list, zip(*col_vals)))

            # Remove any rows that are all empty (from padding)
            def row_has_core(row):
                return any(cell != "" for cell in row)
            second_table_text = [row for row in second_table_text if row_has_core(row)]
            second_table_vals = [row for row in second_table_vals if row_has_core([f"{CORE_LIST[i*n_cols+j]}" if j < len(second_table_text[0]) else "" for j in range(n_cols)])]

            # --- Plot both tables vertically ---
            # Transpose the first table for third display
            transposed_text = df_table.values.T
            transposed_vals = np.array(val_rows).T.tolist()

            # --- Plot all three tables vertically ---
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3,
                figsize=(12, 4 + n_rows * 0.6 + len(df_table) * 0.5)
            )

            ax1.axis("off")
            ax2.axis("off")
            # Transposed first table (displayed last)
            ax3.axis("off")
            tbl3 = ax3.table(cellText=transposed_text, loc="center", cellLoc="center")
            for i, row in enumerate(transposed_vals):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)
                    tbl3[i, j].set_facecolor(color)
            tbl3.auto_set_font_size(False)
            tbl3.set_fontsize(10)
            tbl3.scale(1, 1.5)

            # First table (original, heatmap-style)
            tbl1 = ax1.table(cellText=df_table.values, loc="center", cellLoc="center")
            for i, row in enumerate(val_rows):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)
                    tbl1[i, j].set_facecolor(color)
            tbl1.auto_set_font_size(False)
            tbl1.set_fontsize(10)
            tbl1.scale(1, 1.5)

            # Second table (6 cores per column, no column labels)
            tbl2 = ax2.table(cellText=second_table_text, loc="center", cellLoc="center")
            for i, row in enumerate(second_table_vals):
                for j, val in enumerate(row):
                    label = second_table_text[i][j]
                    if label and pd.notnull(val):
                        color = cmap(norm(val))
                    else:
                        color = (1, 1, 1, 1)
                    tbl2[i, j].set_facecolor(color)
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(10)
            tbl2.scale(1, 1.5)

            # Add a colorbar to the right of the second table
            for ax in [ax1, ax2, ax3]:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
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