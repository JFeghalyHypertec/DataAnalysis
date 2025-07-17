import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(25)]  # Adjust if you have more/less cores

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
    st.subheader("🌡️ Average Temperature Table by Core Group")

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

            # Group cores: 0-5, 6-11, 12-17, 18-23, 24
            groups = [list(range(i, min(i+6, 25))) for i in range(0, 25, 6)]
            table_data = []
            for group in groups:
                row = []
                for core in group:
                    core_name = f"Core {core}"
                    row.append(avg_temps.get(core_name, None))
                # Pad row to length 6 for display consistency
                while len(row) < 6:
                    row.append(None)
                table_data.append(row)

            # Prepare column headers
            col_headers = [f"Core {i}" for i in range(6)]
            df_table = pd.DataFrame(table_data, columns=col_headers)

            # Display as image
            fig, ax = plt.subplots(figsize=(8, 2 + len(df_table)*0.5))
            ax.axis('off')
            tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            plt.tight_layout()

            st.pyplot(fig)

            # Save as image button
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