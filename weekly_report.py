import os
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime, timedelta
from scipy.stats import zscore
from db_connector import fetch_evaluation_data

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    df = fetch_evaluation_data()
    if df.empty:
        raise ValueError("No data returned from database")
    
    avis_map = {
        'tres satisfait': 4,
        'satisfait': 3,
        'peu satisfait': 2,
        'pas du tout satisfait': 1
    }
    df['avis'] = df['avis'].map(avis_map)
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'])
    df['day'] = df['datetime'].dt.date
    
    df['day'] = pd.to_datetime(df['day']).dt.normalize()
    current_date = datetime.now().date()
    start_date = current_date - timedelta(days=6)
    df = df[df['day'].dt.date >= start_date]
    
    return df

def plot_daily_average(df):
    daily_avg = df.groupby('day')['avis'].mean()
    plt.figure(figsize=(6, 4))
    daily_avg.plot(marker='o')
    plt.title("Daily Average Satisfaction (Last Week)")
    plt.xlabel("Day")
    plt.ylabel("Average Score")
    plt.ylim(1, 4)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "daily_avg.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_center_average(df):
    center_avg = df.groupby('code_centre')['avis'].mean()
    plt.figure(figsize=(6, 4))
    center_avg.plot(kind='bar', color='skyblue')
    plt.title("Center-wise Average Satisfaction (Last Week)")
    plt.xlabel("Center")
    plt.ylabel("Average Score")
    plt.ylim(1, 4)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "center_avg.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_distribution(df):
    plt.figure(figsize=(6, 4))
    df['avis'].plot(kind='hist', bins=4, rwidth=0.8, color='salmon')
    plt.title("Satisfaction Distribution (Last Week)")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "distribution.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_heatmap(df):
    heatmap_data = df.groupby(['day', 'code_centre'])['avis'].mean().unstack()
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Avg Satisfaction')
    plt.xticks(ticks=range(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45)
    plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index)
    plt.title("Heatmap of Satisfaction by Day and Center (Last Week)")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "heatmap.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_pie(df):
    counts = df['avis'].value_counts().sort_index()
    label_map = {4: "Très Satisfait", 3: "Satisfait", 2: "Peu Satisfait", 1: "Pas du tout Satisfait"}
    labels = [label_map.get(val, str(val)) for val in counts.index]
    plt.figure(figsize=(6, 4))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des avis (Last Week)")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "pie_chart.png")
    plt.savefig(path)
    plt.close()
    return path

def detect_anomalies(df, threshold=2):
    daily_avg = df.groupby(['code_centre', 'day'])['avis'].mean().reset_index()
    daily_avg.rename(columns={'avis': 'avg_score'}, inplace=True)
    daily_avg['zscore'] = daily_avg.groupby('code_centre')['avg_score'].transform(zscore)
    daily_avg['anomaly'] = daily_avg['zscore'].abs() > threshold
    anomalies = daily_avg[daily_avg['anomaly']]
    return anomalies[['code_centre', 'day', 'avg_score']]

def create_pdf(plots, anomalies_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Weekly Satisfaction Report (Last 7 Days)", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)

    for title, img_path in plots:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(img_path, w=180)
        pdf.ln(5)

    if not anomalies_df.empty:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detected Anomalies (Last Week)", ln=True, align='L')
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        for _, row in anomalies_df.iterrows():
            pdf.cell(0, 10, f"Center: {row['code_centre']} | Date: {row['day']} | Score: {row['avg_score']:.2f}", ln=True)

    report_path = os.path.join(REPORT_DIR, "weekly_report.pdf")
    pdf.output(report_path)
    return report_path

def generate_weekly_report():
    try:
        df = load_data()
        if df.empty:
            return None
            
        plots = [
            ("1. Daily Average Satisfaction", plot_daily_average(df)),
            ("2. Center-wise Average Satisfaction", plot_center_average(df)),
            ("3. Satisfaction Distribution", plot_distribution(df)),
            ("4. Heatmap of Satisfaction by Day and Center", plot_heatmap(df)),
            ("5. Pie Chart of Satisfaction Levels", plot_pie(df))
        ]
        anomalies_df = detect_anomalies(df, threshold=1.5)
        report_path = create_pdf(plots, anomalies_df)
        return report_path
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

if __name__ == "__main__":
    generate_weekly_report()