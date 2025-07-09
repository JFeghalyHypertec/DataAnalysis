def run_graphs_plot():    
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    START_ROW = 3

    CPU_TIME_SERIE_PLOT = ["CPU Package", "Core Max"]

    CORE_LIST = [
        "Core 0", "Core 1", "Core 2", "Core 3", "Core 4", "Core 5", "Core 6", "Core 7",
        "Core 8", "Core 9", "Core 10", "Core 11", "Core 12", "Core 13", "Core 14", "Core 15",
        "Core 16", "Core 17", "Core 18", "Core 19", "Core 20", "Core 21", "Core 22", "Core 23",
        "Core 24", "Core 25"
    ]

    def select_files():
        """
        Allows user to select one file at a time.
        Repeats the selection process based on user confirmation.
        Returns a list of tuples: [(DataFrame, filename), ...]
        """
        root = tk.Tk()
        root.withdraw()

        messagebox.showinfo("Select file", "Please select one CSV, XLS, or XLSX file.")
        data_list = []

        while True:
            file_path = filedialog.askopenfilename(
                title="Select a CSV or Excel file",
                filetypes=[
                    ("CSV and Excel files", "*.csv *.xls *.xlsx"),
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xls *.xlsx")
                ]
            )

            if not file_path:
                messagebox.showinfo("No selection", "No file selected.")
                return data_list
            else:
                ext = Path(file_path).suffix.lower()
                file_name = Path(file_path).name

                if ext not in ['.csv', '.xls', '.xlsx']:
                    messagebox.showwarning("Invalid file", f"{file_name} is not a supported file type.")
                else:
                    try:
                        if ext == '.csv':
                            df = pd.read_csv(file_path, header=None, low_memory=False)
                        else:
                            df = pd.read_excel(file_path, header=None)
                        data_list.append((df, file_name))
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load file {file_name}:\n{str(e)}")

            retry = messagebox.askyesno("Select another file", "Do you want to select another file?")
            if not retry:
                break

        return data_list

    def plot_time_series(time, temperature, label, file_name, save_path=None):
        """
        Plots or saves a time series chart.

        Args:
            time (pd.Series): Time values.
            temperature (pd.Series): Temperature or parameter values.
            label (str): Title and y-axis label.
            save_path (str or None): If provided, saves the plot to this path. Otherwise, displays it.
        """
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(time, temperature, marker='o')
        ax.set_xlabel("Time (s)", color='black')
        ax.set_ylabel("Temperature (°C)", color='black')
        ax.set_title(f"Time Series of {label} in file " + file_name, color='black')

        ax.grid(False)
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.locator_params(axis='x', nbins=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor='white')
        else:
            plt.show()

        plt.close(fig)

    def get_time_series(df, parameter, file_name):
        """
        Displays and optionally saves a time series plot for a specific parameter.

        Args:
            df (pd.DataFrame): Input data.
            parameter (str): Parameter to plot (must exist in row 1 of a column).
        """
        # Hide the root Tk window
        root = tk.Tk()
        root.withdraw()

        for col in df.columns:
            if df.iloc[1, col] == parameter:
                # Convert time and temperature to numeric, handle invalids
                time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
                temperature = pd.to_numeric(df.iloc[START_ROW:, col], errors='coerce')
                label = df.iloc[1, col]

                # Create DataFrame and clean it
                plot_data = pd.DataFrame({
                    'time': time,
                    'value': temperature
                }).dropna()
                plot_data = plot_data.iloc[2:]
                #plot_data['value'] = remove_outliers_iqr(plot_data['value'])
                # Sort by time
                plot_data = plot_data.sort_values(by='time')

                # Plot
                plot_time_series(plot_data['time'], plot_data['value'], label, file_name)

                # Ask user if they want to save
                save = messagebox.askyesno("Save Plot", f"Do you want to save the time series plot of {label}?")
                if save:
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".png",
                        filetypes=[("PNG files", "*.png")],
                        title="Save Plot As",
                        initialfile=label.replace(" ", "_") + ".png"
                    )
                    if file_path:
                        plot_time_series(plot_data['time'], plot_data['value'], label, save_path=file_path)
                        messagebox.showinfo("Saved", f"Plot saved to:\n{file_path}")
                return

    def plot_core_temperature_dominance(df, file_name):
        # Map core labels to column indices
        core_indices = {df.iloc[1, i]: i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST}

        if not core_indices:
            print("No core columns found.")
            return

        # Initialize counts
        core_counts = {core: 0 for core in CORE_LIST}

        for row_idx in range(START_ROW, df.shape[0]):
            temps = {}
            for core, col_idx in core_indices.items():
                try:
                    value = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
                    if not np.isnan(value):
                        temps[core] = value
                except:
                    continue

            if temps:
                max_temp = max(temps.values())
                for core in CORE_LIST:
                    if temps.get(core) == max_temp:
                        core_counts[core] += 1
                        break

        filtered_counts = {k: v for k, v in core_counts.items() if v > 0}
        labels = list(filtered_counts.keys())
        counts = list(filtered_counts.values())
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]

        # Plot histogram
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(labels, counts)
        ax1.set_xticklabels(labels, rotation=90)
        ax1.set_ylabel("Times Core was Max")
        ax1.set_title("Core Max Temperature Count (Histogram) of " + file_name)
        plt.tight_layout()
        plt.show()

        # Ask to save histogram
        root = tk.Tk()
        root.withdraw()
        save_hist = messagebox.askyesno("Save Histogram", "Do you want to save the histogram plot?")
        if save_hist:
            hist_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                title="Save Histogram As",
                initialfile="core_temperature_histogram.png"
            )
            if hist_path:
                fig1.savefig(hist_path)

        # Plot pie chart
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(
            percentages,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False
        )
        ax2.set_title("Core Max Temperature Percentage Dominance (Pie Chart) of " + file_name)
        ax2.axis("equal")
        plt.tight_layout()
        plt.show()

        # Ask to save pie chart
        save_pie = messagebox.askyesno("Save Pie Chart", "Do you want to save the pie chart?")
        if save_pie:
            pie_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                title="Save Pie Chart As",
                initialfile="core_temperature_pie_chart.png"
            )
            if pie_path:
                fig2.savefig(pie_path)

    data_files = select_files()
    if not data_files:
        exit()

    for data in data_files:
        df, file_name = data[0], data[1]
        for parameter in CPU_TIME_SERIE_PLOT:
            get_time_series(df, parameter, file_name)
            time.sleep(1)
    
        plot_core_temperature_dominance(df, file_name)
        time.sleep(1)
        
if __name__ == "__main__":
    run_graphs_plot()