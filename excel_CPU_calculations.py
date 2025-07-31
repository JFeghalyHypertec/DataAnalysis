"""
How to add a new Paramter to the returned CSV file:
1- Add in PARAMETERS list the name of the new parameter you want to add, Example: Water Temperature
2- Add in the UNITS list the unit of the new parameter you want to add at the SAME index as the parameter added in PARAMETERS. Example: °C
3- Add the name of the new parameter as a global variable and name it the same way it was named in the excel file, Example: WATER_TEMP
4- Navigate to the calculate_metrics(df) function and in the values list, at the SAME exact index as where the unit and the parameter
   were added in the UNITS and PARAMETERS list write: safe_round(function(df, list_targeted or string_targeted), number_decimals), function represents the metric you want to calculate
   list_targeted represent the list or string that the program will loop through (example: CORE_LIST or WATER_TEMP), number_decimals: the number of decimal (1 by default).
   Example: safe_round(get_average_per_parameter(df, WATER_TEMP))
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

START_ROW = 3

# Same constants as before...
PARAMETERS = [
    "CPU Package Temperature", "Core Max Temperature", "Core Average", "Max CPU Package Temperature",
    "Max Core Max Temperature", "StDev CPU Package Temperature", "StDev Core Max Temperature", 
    "Average StDev Population Core Temperature", "CPU Package Power", "Total Power In","Total Power Out", 
    "Average Core Clock", "Average Core V1D",  "TR1 Temperature", "Average SYS FAN", "Pump Fan", 
    "Water Flow", "Water Temperature Out", "Water Temperature In"
]
UNITS = [
    "°C", "°C", "°C", "°C", "°C", "°C", "°C", "°C", "W", "W", "W",
    "MHz", "V", "°C", "rpm", "rpm", "l/h", "°C", "°C"
]

CORE_CLOCK_LIST = [f"Core {i} Clock (perf #{i%27})" for i in range(26)]
FAN_LIST = [f"SYS_FAN{i+1} (Fan)" for i in range(8)]
CORE_LIST = [f"Core {i}" for i in range(26)]
CORE_VID_LIST = [f"Core {i} VID" for i in range(26)]
PSU_POWER_IN_LIST = ["PSU1 Power In (Power Supply)", "PSU2 Power In (Power Supply)"]
PSU_POWER_OUT_LIST = ["PSU1 Power Out (Power Supply)", "PSU2 Power Out (Power Supply)"]

TR1_TEMP = "TR1 Temperature (System Board)"
PUMP_FAN = "PUMP_FAN (Fan)"
CPU_PACKAGE = "CPU Package"
CPU_POWER = "CPU Package Power"
CORE_MAX = "Core Max"
WATER_FLOW = "Water Flow"
WATER_TEMP_IN = "Water Temperature"
WATER_TEMP_OUT = "External Temperature"

def safe_round(value, ndigits=1, as_int=False):
    if value is None: return None
    r = round(value, ndigits)
    return int(r) if as_int and ndigits == 0 else r

def get_col(df, name):
    try:
        return next(i for i in range(df.shape[1]) if df.iloc[1, i] == name)
    except StopIteration:
        return None

def get_numeric_col(df, name):
    col_idx = get_col(df, name)
    if col_idx is None: return pd.Series(dtype=float)
    values = pd.to_numeric(df.iloc[START_ROW:, col_idx], errors='coerce')
    return values[(values != 0) & (~values.isna())]

def avg_of_avgs(df, targets):
    cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in targets]
    if not cols: return None
    data = df.iloc[START_ROW:, cols].apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
    return data.mean(axis=1).mean()

def stddev_of_population(df, targets):
    cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in targets]
    if not cols: return None
    data = df.iloc[START_ROW:, cols].apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
    return data.std(axis=1, ddof=0).mean()

def total_avg(df, group):
    values = [get_numeric_col(df, g).mean() for g in group]
    return sum(v for v in values if v is not None)

def calculate_metrics(df):
    return [
        safe_round(get_numeric_col(df, CPU_PACKAGE).mean()),
        safe_round(get_numeric_col(df, CORE_MAX).mean()),
        safe_round(avg_of_avgs(df, CORE_LIST)),
        safe_round(get_numeric_col(df, CPU_PACKAGE).max(), 0),
        safe_round(get_numeric_col(df, CORE_MAX).max(), 0),
        safe_round(get_numeric_col(df, CPU_PACKAGE).std()),
        safe_round(get_numeric_col(df, CORE_MAX).std()),
        safe_round(stddev_of_population(df, CORE_LIST)),
        safe_round(get_numeric_col(df, CPU_POWER).mean()),
        safe_round(total_avg(df, PSU_POWER_IN_LIST)),
        safe_round(total_avg(df, PSU_POWER_OUT_LIST)),
        safe_round(avg_of_avgs(df, CORE_CLOCK_LIST), 0),
        safe_round(avg_of_avgs(df, CORE_VID_LIST)),
        safe_round(get_numeric_col(df, TR1_TEMP).mean()),
        safe_round(avg_of_avgs(df, FAN_LIST), 0),
        safe_round(get_numeric_col(df, PUMP_FAN).mean(), 0),
        safe_round(get_numeric_col(df, WATER_FLOW).mean()),
        safe_round(get_numeric_col(df, WATER_TEMP_IN).mean()),
        safe_round(get_numeric_col(df, WATER_TEMP_OUT).mean())
    ]

def run_excel_CPU_calculations():
    st.title("📊 Excel CPU Performance Metrics Analyzer")
    uploaded_files = st.file_uploader("Upload one or more CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)
    
    if not uploaded_files:
        return
    
    if uploaded_files:
        combined_rows = [["File Name"] + PARAMETERS, [""] + UNITS]

        for uploaded in uploaded_files:
            file_name = uploaded.name
            ext = file_name.split(".")[-1].lower()

            try:
                if ext == "csv":
                    df = pd.read_csv(uploaded, header=None, low_memory=False)
                else:
                    df = pd.read_excel(uploaded, header=None)
            except Exception as e:
                st.error(f"Failed to load {file_name}: {e}")
                continue

            values = calculate_metrics(df)
            file_name = file_name.replace(".csv", "").replace(".xls", "").replace(".xlsx", "")
            combined_rows.append([file_name] + values)

        # Display results
        results = pd.DataFrame(combined_rows[2:], columns=combined_rows[0])
        st.subheader("📋 Results Preview")
        st.dataframe(results)

        # Convert to Excel for download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(combined_rows).to_excel(writer, index=False, header=False)
        output.seek(0)
        f_name = "results.xlsx"
        st.download_button("📥 Download Excel File", data=output, file_name=f_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
