import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from io import BytesIO
import numpy as np

# -----------------------------
# Constants
# -----------------------------
START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]  # Expected 26 cores: Core 0..Core 25


# -----------------------------
# Data extraction
# -----------------------------
def extract_core_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-core temperatures from raw OCCT dataframe.

    - Detects core columns by header row (index 1) matching CORE_LIST.
    - Parses first column as time (seconds).
    - Normalizes time to start at 0 (uses the first valid timestamp).
    """
    # Locate core columns by their header names in row 1
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")

    # Parse time column (first column) and normalize to start at 0
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    first_valid = time.first_valid_index()
    if first_valid is None:
        raise ValueError("Time column has no valid numeric values.")
    t0 = time.loc[first_valid]
    time = time - t0

    # Coerce core values to numeric; keep NaN if parsing fails
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data


# -----------------------------
# Main UI
# -----------------------------
def run_display_core_avg_table():
    """
    Streamlit workflow:
    - Upload one or more CSV files
    - For each file: extract cores → compute avg per core → render two color-coded tables:
        1) Row-major (6 per row)
        2) Column-major (6 columns)
    - Each cell shows "Core X = YY.YY°C" and is colorized by its value.
    - Download the combined figure as PNG.
    """
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
            # Load file and extract core timeseries
            df = pd.read_csv(uploaded_file, header=None)
            core_df = extract_core_data(df)

            # Average temperature per core over the whole run
            avg_temps = core_df.mean()

            # Build lists aligned to CORE_LIST (handles missing cores with NaN)
            values = [avg_temps.get(core) for core in CORE_LIST]
            texts = [
                f"{core} = {val:.2f}°C" if pd.notnull(val) else f"{core} = N/A"
                for core, val in zip(CORE_LIST, values)
            ]

            # ---------- Table 1: 6-per-row ----------
            text_rows = [texts[i:i + 6] for i in range(0, len(texts), 6)]
            val_rows = [values[i:i + 6] for i in range(0, len(values), 6)]

            # Pad last row to width 6
            if text_rows:
                while len(text_rows[-1]) < 6:
                    text_rows[-1].append("")
                    val_rows[-1].append(np.nan)

            df_table = pd.DataFrame(text_rows)

            # Normalize values for coloring; handle all-NaN or constant arrays
            flat_vals = [v for row in val_rows for v in row if not pd.isna(v)]
            if len(flat_vals) == 0:
                # No numeric values: use a dummy range so colorbars render
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = float(np.min(flat_vals)), float(np.max(flat_vals))
                if vmin == vmax:  # avoid divide-by-zero in Normalize
                    vmin -= 0.5
                    vmax += 0.5

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("coolwarm")

            # Figure layout: table + colorbar, then transposed table + colorbar
            fig, (ax1, cax1, ax2, cax2) = plt.subplots(
                nrows=4,
                figsize=(13, 5 + len(df_table) * 0.6),
                gridspec_kw={"height_ratios": [1, 0.05, 1, 0.05]}
            )
            ax1.axis("off")
            ax2.axis("off")

            # ---- Render Table 1 (row-major) ----
            tbl1 = ax1.table(cellText=df_table.values, loc="center", cellLoc="center")
            for i, row in enumerate(val_rows):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)  # white for N/A
                    tbl1[i, j].set_facecolor(color)
            tbl1.auto_set_font_size(False)
            tbl1.set_fontsize(10)
            tbl1.scale(1, 1.5)

            # Colorbar for table 1
            sm1 = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm1.set_array([])
            cbar1 = fig.colorbar(sm1, cax=cax1, orientation='horizontal')
            cbar1.set_label("Temperature (°C)")

            # ---------- Table 2: transposed (6 columns) ----------
            n_cols = 6
            n_rows = int(np.ceil(len(CORE_LIST) / n_cols))
            transposed_vals_cols = []
            transposed_texts_cols = []

            for col in range(n_cols):
                start = col * n_rows
                end = (col + 1) * n_rows
                sub_vals = values[start:end]
                sub_texts = texts[start:end]
                # Pad to equal length
                while len(sub_vals) < n_rows:
                    sub_vals.append(np.nan)
                    sub_texts.append("")
                transposed_vals_cols.append(sub_vals)
                transposed_texts_cols.append(sub_texts)

            # Convert column-wise lists to row-wise for mpl.table
            transposed_vals = np.array(transposed_vals_cols).T.tolist()
            transposed_texts = np.array(transposed_texts_cols).T.tolist()

            tbl2 = ax2.table(cellText=transposed_texts, loc="center", cellLoc="center")
            for i, row in enumerate(transposed_vals):
                for j, val in enumerate(row):
                    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 1)
                    tbl2[i, j].set_facecolor(color)
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(10)
            tbl2.scale(1, 1.5)

            # Colorbar for table 2 (same scale)
            sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm2.set_array([])
            cbar2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
            cbar2.set_label("Temperature (°C)")

            plt.tight_layout()
            st.pyplot(fig)

            # -------- Download PNG --------
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            st.download_button(
                label=f"💾 Save Table as PNG ({uploaded_file.name})",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name}_core_avg_table_with_transpose.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❗ Error processing {uploaded_file.name}: {e}")
            st.exception(e)
