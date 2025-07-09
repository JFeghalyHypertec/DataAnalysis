def run_core_heatmap_plot():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tkinter import filedialog, messagebox, Tk
    import matplotlib.gridspec as gridspec

    START_ROW = 3
    CORE_LIST = [f"Core {i}" for i in range(26)]

    def select_file():
        root = Tk()
        root.withdraw()
        return filedialog.askopenfilename(
            title="Select a CSV or Excel file",
            filetypes=[("CSV and Excel files", "*.csv *.xls *.xlsx")]
        )

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
        if not core_cols:
            raise ValueError("No core temperature columns found.")

        time = pd.to_numeric(df.iloc[START_ROW:, time_col], errors='coerce')
        core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
        core_data.index = time
        core_data.columns = [df.iloc[1, i] for i in core_cols]

        return core_data

    def plot_heatmap(core_df):
        averages = core_df[core_df != 0].mean()
        overall_avg = averages.mean()

        fig = plt.figure(figsize=(16, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[4, 1])

        # Heatmap
        ax0 = fig.add_subplot(spec[0])
        sns.heatmap(core_df.T, cmap="coolwarm", cbar_kws={'label': 'Temperature (°C)'}, ax=ax0)
        ax0.set_title("Core Temperatures Over Time (Heatmap)")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("CPU Cores")

        # Bar plot
        ax1 = fig.add_subplot(spec[1])
        bars = ax1.barh(averages.index, averages.values, color='gray')
        ax1.set_title("Avg Temp per Core")
        ax1.set_xlim(averages.min() - 5, averages.max() + 5)
        ax1.set_xlabel("°C")

        # Add value labels to each bar
        for bar, value in zip(bars, averages.values):
            ax1.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}°C",
                    va='center', ha='left', fontsize=9)

        # Add overall average label at the top
        ax1.text(
            0.5, 1.05,
            f"Overall Avg Temp: {overall_avg:.1f}°C",
            ha='center', va='center',
            transform=ax1.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightyellow')
        )

        plt.tight_layout()
        plt.show()

        root = Tk()
        root.withdraw()
        save = messagebox.askyesno("Save Heatmap", "Do you want to save the heatmap with avg temperatures?")
        if save:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                title="Save Heatmap As",
                initialfile="core_temperature_heatmap.png"
            )
            if file_path:
                fig.savefig(file_path)
                messagebox.showinfo("Saved", f"Heatmap saved to:\n{file_path}")
        plt.close(fig)


    file_path = select_file()
    if not file_path:
        print("No file selected.")
        exit()

    try:
        df = load_file(file_path)
        core_df = extract_core_data(df)
        plot_heatmap(core_df)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate heatmap:\n{str(e)}")
        
if __name__ == "__main__":
    run_core_heatmap_plot()