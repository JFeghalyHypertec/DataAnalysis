"""
How to add a new Paramter to the returned CSV file:
1- Add in PARAMETERS list the name of the new parameter you want to add, Example: Water Temperature
2- Add in the UNITS list the unit of the new parameter you want to add at the SAME index as the parameter added in PARAMETERS. Example: °C
3- Navigate to the calculate_metrics(df) function and in the values list, at the SAME exact index as where the unit and the parameter
   were added in the UNITS and PARAMETERS list write: safe_round(function(df, list_targeted or string_targeted), number_decimals), function represents the metric you want to calculate
   list_targeted represent the list or string that the program will loop through (example: CORE_LIST or WATER_TEMP), number_decimals: the number of decimal (1 by default).
   Example: safe_round(get_average_per_parameter(df, WATER_TEMP))
   
"""
def run_excel_CPU_calculations():
    """
    Main function to run the CPU calculations script.
    This function is intended to be called when the script is executed directly.
    """

    import tkinter as tk
    from tkinter import filedialog, messagebox
    from pathlib import Path
    import pandas as pd # type: ignore
    import os
    import platform
    import subprocess

    START_ROW = 3

    FAN_LIST = [
        "SYS_FAN1 (Fan)", "SYS_FAN2 (Fan)", "SYS_FAN3 (Fan)", "SYS_FAN4 (Fan)",
        "SYS_FAN5 (Fan)", "SYS_FAN6 (Fan)", "SYS_FAN7 (Fan)", "SYS_FAN8 (Fan)"
    ]

    PARAMETERS = [
        "CPU Package Temperature", "Core Max Temperature", "Core Average", "Max CPU Package Temperature",
        "Max Core Max Temperature", "StDev CPU Package Temperature", "StDev Core Max Temperature", 
        "Average StDev Population Core Temperature", "CPU Package Power", "Total Power In","Total Power Out", 
        "Average Core Clock", "Average Core V1D",  "TR1 Temperature", "Average SYS FAN", "Pump Fan", "Water Flow", "Water Temperature"
    ]

    CORE_CLOCK_LIST = [
        "Core 0 Clock (perf #20)",
        "Core 1 Clock (perf #5)",
        "Core 2 Clock (perf #17)",
        "Core 3 Clock (perf #8)",
        "Core 4 Clock (perf #15)",
        "Core 5 Clock (perf #9)",
        "Core 6 Clock (perf #13)",
        "Core 7 Clock (perf #10)",
        "Core 8 Clock (perf #16)",
        "Core 9 Clock (perf #18)",
        "Core 10 Clock (perf #6)",
        "Core 11 Clock (perf #19)",
        "Core 12 Clock (perf #11)",
        "Core 13 Clock (perf #14)",
        "Core 14 Clock (perf #1)",
        "Core 15 Clock (perf #21)",
        "Core 16 Clock (perf #23)",
        "Core 17 Clock (perf #22)",
        "Core 18 Clock (perf #24)",
        "Core 19 Clock (perf #25)",
        "Core 20 Clock (perf #2)",
        "Core 21 Clock (perf #12)",
        "Core 22 Clock (perf #26)",
        "Core 23 Clock (perf #3)",
        "Core 24 Clock (perf #4)",
        "Core 25 Clock (perf #7)"
    ]

    PSU_POWER_IN_LIST = ["PSU1 Power In (Power Supply)", "PSU2 Power In (Power Supply)"]
    PSU_POWER_OUT_LIST = ["PSU1 Power Out (Power Supply)", "PSU2 Power Out (Power Supply)"]

    CORE_LIST = [
        "Core 0", "Core 1", "Core 2", "Core 3", "Core 4", "Core 5", "Core 6", "Core 7",
        "Core 8", "Core 9", "Core 10", "Core 11", "Core 12", "Core 13", "Core 14", "Core 15",
        "Core 16", "Core 17", "Core 18", "Core 19", "Core 20", "Core 21", "Core 22", "Core 23",
        "Core 24", "Core 25"
    ]

    CORE_VID_LIST = [
        "Core 0 VID", "Core 1 VID", "Core 2 VID", "Core 3 VID", "Core 4 VID", "Core 5 VID",
        "Core 6 VID", "Core 7 VID", "Core 8 VID", "Core 9 VID", "Core 10 VID", "Core 11 VID",
        "Core 12 VID", "Core 13 VID", "Core 14 VID", "Core 15 VID", "Core 16 VID", "Core 17 VID",
        "Core 18 VID", "Core 19 VID", "Core 20 VID", "Core 21 VID", "Core 22 VID", "Core 23 VID",
        "Core 24 VID", "Core 25 VID"
    ]

    UNITS = [
        "°C",  # CPU Package Temperature
        "°C",  # Core Max Temperature
        "°C",  # Core Average
        "°C",  # Max CPU Package Temperature
        "°C",  # Max Core Max Temperature
        "°C",  # StDev CPU Package Temperature
        "°C",  # StDev Core Max Temperature
        "°C",  # Avg StDev Population CPU average
        "W",   # CPU Package Power
        "W",   # Total Power In
        "W",   # Total Power Out
        "MHz", # Average Core Clock
        "V",   # Average Core V1D
        "°C",  # TR1 Temperature
        "rpm", # Average SYS FAN
        "rpm", # Pump Fan
        "l/h", # Water Flow
        "°C",  # Water Temperature
    ]

    TR1_TEMP = "TR1 Temperature (System Board)"
    PUMP_FAN = "PUMP_FAN (Fan)"
    CPU_PACKAGE = "CPU Package"
    CPU_POWER = "CPU Package Power"
    CORE_MAX = "Core Max"
    WATER_FLOW = "Water Flow"
    WATER_TEMP = "Water Temperature"

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

    def get_average_of_averages_per_time(df, elements):
        columns = [i for i in range(df.shape[1]) if df.iloc[1, i] in elements]
        if not columns:
            return None

        data = df.iloc[START_ROW:, columns].apply(pd.to_numeric, errors='coerce')
        data = data.mask(data == 0)
        avg_per_row = data.mean(axis=1, skipna=True)
        return avg_per_row.mean(skipna=True)

    def get_average_stdev_population_per_time(df, elements):
        columns = [i for i in range(df.shape[1]) if df.iloc[1, i] in elements]
        if not columns:
            return None

        data = df.iloc[START_ROW:, columns].apply(pd.to_numeric, errors='coerce')
        data = data.mask(data == 0)
        stddev_per_row = data.std(axis=1, skipna=True, ddof=0)
        avg_stddev = stddev_per_row.mean(skipna=True)
        return avg_stddev

    def get_average_per_parameter(df, parameter):
        try:
            col_idx = next(i for i in range(df.shape[1]) if df.iloc[1, i] == parameter)
        except StopIteration:
            return None

        values = pd.to_numeric(df.iloc[START_ROW:, col_idx], errors='coerce')
        values = values[(values != 0) & (~values.isna())]  # Exclude 0s and NaNs
        return values.mean() if not values.empty else None

    def get_maximum_per_parameter(df, parameter):
        try:
            col_idx = next(i for i in range(df.shape[1]) if df.iloc[1, i] == parameter)
        except StopIteration:
            return None

        values = pd.to_numeric(df.iloc[START_ROW:, col_idx], errors='coerce')
        return values.max(skipna=True)

    def get_standard_deviation_per_parameter(df, parameter):
        try:
            col_idx = next(i for i in range(df.shape[1]) if df.iloc[1, i] == parameter)
        except StopIteration:
            return None

        values = pd.to_numeric(df.iloc[START_ROW:, col_idx], errors='coerce')
        values = values[(values != 0) & (~values.isna())]  # Exclude 0s and NaNs
        return values.std(ddof=1) if not values.empty else None


    def get_average_of_total_power(df, parameters):
        averages = [get_average_per_parameter(df, p) for p in parameters]
        valid = [a for a in averages if a is not None]
        return sum(valid) if valid else None          # or 0 if you prefer


    def calculate_metrics(df):
        """
        Computes a list of key hardware performance metrics from the input DataFrame.

        This function aggregates various statistics such as averages, maximums, and standard deviations 
        for CPU temperature, power, clock speeds, fan speeds, and more. It uses helper functions to extract 
        these values and returns them in a fixed order that aligns with the PARAMETERS and UNITS lists.

        Returns:
            list: A list of rounded metric values (floats or ints), or None where data is unavailable.
        """
        
        def safe_round(value, ndigits=1, as_int=False) -> float:
            """
            Safely round a value to the specified number of digits.
            Returns None if the input is None.

            Args:
                value (float or None): The value to round.
                ndigits (int): Number of decimal places.
                as_int (bool): If True and ndigits is 0, return an integer.

            Returns:
                float or int or None
            """
            if value is None:
                return None
            rounded = round(value, ndigits)
            return int(rounded) if as_int and ndigits == 0 else rounded

        # Precompute averages for values used multiple times
        cpu_clock_avg = get_average_of_averages_per_time(df, CORE_CLOCK_LIST)
        fan_avg = get_average_of_averages_per_time(df, FAN_LIST)
        pump_avg = get_average_per_parameter(df, PUMP_FAN)
        # Compute and collect all metrics in the expected order
        values = [
            safe_round(get_average_per_parameter(df, CPU_PACKAGE)),                       # Avg CPU Temp
            safe_round(get_average_per_parameter(df, CORE_MAX)),                          # Avg Core Max Temp
            safe_round(get_average_of_averages_per_time(df, CORE_LIST)),                  # Avg Core Temp
            safe_round(get_maximum_per_parameter(df, CPU_PACKAGE), 0),                    # Max CPU Temp
            safe_round(get_maximum_per_parameter(df, CORE_MAX), 0),                       # Max Core Max Temp
            safe_round(get_standard_deviation_per_parameter(df, CPU_PACKAGE)),            # Std Dev CPU Temp
            safe_round(get_standard_deviation_per_parameter(df, CORE_MAX)),               # Std Dev Core Max Temp
            safe_round(get_average_stdev_population_per_time(df, CORE_LIST)),             # Avg Std Dev population
            safe_round(get_average_per_parameter(df, CPU_POWER)),                         # Avg CPU Power
            safe_round(get_average_of_total_power(df, PSU_POWER_IN_LIST)),                # Avg Power In
            safe_round(get_average_of_total_power(df, PSU_POWER_OUT_LIST)),               # Avg Power Out
            safe_round(cpu_clock_avg, 0),                                                 # Avg Core Clock
            safe_round(get_average_of_averages_per_time(df, CORE_VID_LIST)),              # Avg Core VID
            safe_round(get_average_per_parameter(df, TR1_TEMP)),                          # TR1 Temp
            safe_round(fan_avg, 0),                                                       # Avg Fan RPM
            safe_round(pump_avg, 0),                                                      # Pump RPM
            safe_round(get_average_per_parameter(df, WATER_FLOW)),                        # Water Flow l/h avg
            safe_round(get_average_per_parameter(df, WATER_TEMP))                         # Water Temperature avg
        ]
            
        return values

  
    """
    Main execution flow:
    - Prompt user to select a file (CSV/XLS/XLSX).
    - Extract relevant hardware metrics from the file.
    - Ask the user to choose a location to save the results as an Excel file.
    - Save results including parameters, units, and calculated values.
    """

    # Step 1: Prompt the user to select a data file
    data_files = select_files()
    if not data_files:
        exit()

    combined_rows = []
    
    header = ["File Name"] + PARAMETERS
    unit_rows = [""] + UNITS
    combined_rows.append(header)
    combined_rows.append(unit_rows)

    for df, file_name in data_files:
        values = calculate_metrics(df)
        row = [file_name] + values
        combined_rows.append(row)


    # Step 3: Ask the user for a location to save the Excel results
    root = tk.Tk()
    root.withdraw()

    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        title="Save results as",
        initialfile="results.xlsx"
    )

    if not save_path:
        messagebox.showinfo("Canceled", "Save operation was canceled.")
        exit()

    # Step 4: Delete any existing file at the save path to avoid conflicts
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete existing file:\n{save_path}\n\n{str(e)}")
            exit()

    # Step 5: Save results to Excel file
    try:
        results = pd.DataFrame(combined_rows)        
        results.to_excel(save_path, index=False, header=False)
        messagebox.showinfo("Success", f"Combined results saved to:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save Excel file:\n{str(e)}")
        exit()
    
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(save_path)
        elif system == "Darwin":  # macOS
            subprocess.call(["open", save_path])
        else:  # Linux and others
            subprocess.call(["xdg-open", save_path])
    except Exception as e:
        messagebox.showwarning("Warning", f"Could not open the file automatically:\n{str(e)}")
            
if __name__ == "__main__":
    run_excel_CPU_calculations()