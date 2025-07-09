def run_core_heatmap_comparison():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tkinter import filedialog, messagebox, Tk
    import os

    START_ROW = 3
    CORE_LIST = [f"Core {i}" for i in range(26)]

    def ask_for_file(prompt):
        root = Tk()
        root.withdraw()
        messagebox.showinfo("Select File", prompt)
        return filedialog.askopenfilename(
            title=prompt,
            filetypes=[("CSV and Excel files", "*.csv *.xls *.xlsx")]
        )

    def confirm_selection(file_path, label):
        root = Tk()
        root.withdraw()
        return messagebox.askyesno("Confirm File", f"{label} selected:\n\n{file_path}\n\nIs this correct?")

    def load_file(file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path, header=None, low_memory=False)
        elif file_path.endswith((".xls", ".xlsx")):
            return pd.read_excel(file_path, header=None)
        else:
            raise ValueError("Unsupported file format.")

    def extract_core_data(df):
        core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
        time_col = 0
        time = pd.to_numeric(df.iloc[START_ROW:, time_col], errors='coerce')
        core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
        core_data.index = time
        core_data.columns = [df.iloc[1, i] for i in core_cols]

        # Treat 0s as missing (convert to NaN)
        core_data = core_data.replace(0, np.nan)

        # Round time to nearest 60 seconds (bucket per minute)
        core_data['time_bucket'] = (core_data.index // 60) * 60
        grouped = core_data.groupby('time_bucket').mean()

        return grouped.dropna(axis=1, how='all')  # drop cores that are all NaN

    def plot_difference_heatmap(df1, df2, file1_path, file2_path, tolerance_seconds=60):
        # Align by time bucket and core names
        t1 = pd.to_numeric(df1.iloc[-1, 0], errors='coerce')
        t2 = pd.to_numeric(df2.iloc[-1, 0], errors='coerce')

        if pd.isna(t1) or pd.isna(t2):
            messagebox.showerror("Invalid Timestamp", "Could not read the final timestamp from one or both files.")
            exit()

        if abs(t1 - t2) > tolerance_seconds:
            messagebox.showerror(
                "Mismatch Detected",
                f"Last timestamps differ too much:\nFile 1: {t1}\nFile 2: {t2}\n\nDifference: {diff:.2f} seconds"
            )
            exit()
            
        common_index = df1.index.intersection(df2.index)
        common_columns = df1.columns.intersection(df2.columns)

        df1_aligned = df1.loc[common_index, common_columns]
        df2_aligned = df2.loc[common_index, common_columns]
        df_diff = df2_aligned - df1_aligned

        plt.figure(figsize=(14, 6))
        sns.heatmap(df_diff.T, cmap="RdBu", center=0, cbar_kws={'label': 'ΔTemp (°C)'})
        title = f"Difference Heatmap\n({os.path.basename(file2_path)} - {os.path.basename(file1_path)})\nAveraged every 60 seconds"
        plt.title(title)
        plt.xlabel("Time Bucket (s)")
        plt.ylabel("CPU Cores")
        plt.tight_layout()
        plt.show()

        # Ask to save
        root = Tk()
        root.withdraw()
        save = messagebox.askyesno("Save Heatmap", "Do you want to save the difference heatmap?")
        if save:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                title="Save Heatmap As",
                initialfile="difference_heatmap_avg_60s.png"
            )
            if file_path:
                plt.figure(figsize=(14, 6))
                sns.heatmap(df_diff.T, cmap="RdBu", center=0, cbar_kws={'label': 'ΔTemp (°C)'})
                plt.title("Difference Heatmap (File 2 - File 1)\nAveraged every 60 seconds")
                plt.xlabel("Time Bucket (s)")
                plt.ylabel("CPU Cores")
                plt.tight_layout()
                plt.savefig(file_path)
                messagebox.showinfo("Saved", f"Heatmap saved to:\n{file_path}")
                plt.close()

    while True:
        file1 = ask_for_file("Select the FIRST file")
        if not file1: exit()
        if confirm_selection(file1, "First file"):
            break

    while True:
        file2 = ask_for_file("Select the SECOND file")
        if not file2: exit()
        if confirm_selection(file2, "Second file"):
            break

    try:
        df1 = extract_core_data(load_file(file1))
        df2 = extract_core_data(load_file(file2))
        plot_difference_heatmap(df1, df2, file1, file2)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        
if __name__ == "__main__":
    run_core_heatmap_comparison()