import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
LOG_FILE = "output/safety_log.csv"

def plot_safety_timeline():
    print("📊 Generating Analysis Graphs...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Run inference first!")
        return

    # 2. Filter for interesting categories
    # We want to compare "VIOLATION" vs "SAFE" vs "AUTHORIZED"
    def categorize(status):
        if "VIOLATION" in status: return "VIOLATION"
        if "AUTHORIZED" in status or "WORK" in status: return "AUTHORIZED_WORK"
        if "SAFE" in status: return "SAFE_WALK"
        if "VEHICLE" in status: return "VEHICLE"
        return "UNKNOWN"

    df['Category'] = df['Final_Decision'].apply(categorize)
    
    # Remove Unknowns for cleaner graphs
    df = df[df['Category'] != 'UNKNOWN']

    # 3. Create the Plot
    plt.figure(figsize=(12, 6))
    
    # Count events per second (Time Series Analysis)
    # We round timestamps to integers to bin them by second
    df['Time_Bin'] = df['Timestamp_Sec'].astype(int)
    
    sns.histplot(
        data=df, 
        x="Time_Bin", 
        hue="Category", 
        multiple="stack", 
        palette={"VIOLATION": "red", "SAFE_WALK": "green", "AUTHORIZED_WORK": "orange", "VEHICLE": "yellow"},
        binwidth=1
    )

    plt.title("Safety Compliance Over Time", fontsize=16)
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Number of Frames Detected", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the chart
    plt.savefig("output/safety_analysis_chart.png")
    print("✅ Graph saved to output/safety_analysis_chart.png")
    plt.show()

if __name__ == "__main__":
    plot_safety_timeline()