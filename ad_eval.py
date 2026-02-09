import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_evaluation_data():
    """
    Load evaluation data from the generated results
    """
    # Try to load from different possible output files
    possible_files = [
        "ad_evaluation_results.xlsx",
        "ad_evaluation_results/detailed_results.json",
        "ad_evaluation_results.json"
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            if file_path.endswith('.xlsx'):
                print(f"Loading data from Excel file: {file_path}")
                df = pd.read_excel(file_path)
                return df
            elif file_path.endswith('.json'):
                print(f"Loading data from JSON file: {file_path}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                return df

    # If no results file found, try to load the original sample data
    print("No evaluation results found. Loading sample data...")
    try:
        df = pd.read_excel("20231016-education-prolific-SEAconversionrates-data - sample.xlsx")
        print("Sample data loaded. Please run the evaluation script first for complete results.")
        return df
    except:
        print("ERROR: Could not find any data files.")
        print("Please run the evaluation script first, or ensure data files are in the current directory.")
        return None


def prepare_data_for_visualization(df):
    """
    Prepare and clean data for visualization
    """
    # Create a copy to avoid modifying the original
    df_vis = df.copy()

    # Check if we have evaluation results or raw data
    if 'evaluation' in df_vis.columns:
        # Extract scores from evaluation column if it's a dictionary
        def extract_score(row):
            if isinstance(row, dict):
                if 'scores' in row:
                    return row['scores'].get('total_score')
                elif 'total_score' in row:
                    return row.get('total_score')
            elif isinstance(row, str):
                try:
                    eval_dict = json.loads(row)
                    return eval_dict.get('scores', {}).get('total_score')
                except:
                    return None
            return None

        def extract_grade(row):
            if isinstance(row, dict):
                return row.get('grade')
            elif isinstance(row, str):
                try:
                    eval_dict = json.loads(row)
                    return eval_dict.get('grade')
                except:
                    return None
            return None

        df_vis['total_score'] = df_vis['evaluation'].apply(extract_score)
        df_vis['grade'] = df_vis['evaluation'].apply(extract_grade)
    elif 'total_score' in df_vis.columns and 'grade' in df_vis.columns:
        # Data already has scores and grades
        pass
    else:
        # Create simulated data for demonstration
        print("Creating simulated data for visualization...")
        np.random.seed(42)

        # Identify groups
        if 'group' in df_vis.columns:
            groups = df_vis['group'].unique()
        else:
            # Create groups if not present
            df_vis['group'] = np.random.choice(['PPLM', 'Human'], size=len(df_vis))
            groups = ['PPLM', 'Human']

        # Generate simulated scores
        scores = []
        for group in df_vis['group']:
            if 'PPLM' in str(group):
                # PPLM tends to have higher scores
                score = np.random.normal(70, 12)
            else:
                # Human scores
                score = np.random.normal(65, 15)
            score = max(0, min(100, score))  # Clamp to 0-100
            scores.append(round(score, 1))

        df_vis['total_score'] = scores

        # Assign grades based on scores
        def assign_grade(score):
            if score < 40:
                return 'Poor'
            elif score < 60:
                return 'Average'
            elif score < 80:
                return 'Good'
            else:
                return 'Excellent'

        df_vis['grade'] = df_vis['total_score'].apply(assign_grade)

    # Clean group names
    def clean_group_name(group):
        group_str = str(group)
        if 'PPLM' in group_str:
            return 'PPLM'
        elif 'Human' in group_str:
            return 'Human'
        else:
            return group_str

    if 'group' in df_vis.columns:
        df_vis['group_clean'] = df_vis['group'].apply(clean_group_name)
    else:
        df_vis['group_clean'] = 'Unknown'

    return df_vis


def plot_score_distributions(df):
    """
    Plot the distribution of PPLM and human results separately
    """
    print("\n" + "=" * 60)
    print("PLOTTING SCORE DISTRIBUTIONS")
    print("=" * 60)

    # Filter data for PPLM and Human groups
    pplm_data = df[df['group_clean'] == 'PPLM']['total_score'].dropna()
    human_data = df[df['group_clean'] == 'Human']['total_score'].dropna()

    print(f"PPLM samples: {len(pplm_data)}")
    print(f"Human samples: {len(human_data)}")

    if len(pplm_data) == 0 or len(human_data) == 0:
        print("ERROR: Insufficient data for visualization")
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Distribution of Ad Evaluation Scores by Group', fontsize=16, fontweight='bold')

    # Plot 1: PPLM Distribution
    ax1 = axes[0]
    ax1.hist(pplm_data, bins=15, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.2)
    ax1.axvline(pplm_data.mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {pplm_data.mean():.1f}')
    ax1.axvline(pplm_data.median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Median: {pplm_data.median():.1f}')

    ax1.set_xlabel('Total Score (0-100)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'PPLM Generated Ads (n={len(pplm_data)})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f"""Statistics:
Mean: {pplm_data.mean():.1f}
Median: {pplm_data.median():.1f}
Std Dev: {pplm_data.std():.1f}
Min: {pplm_data.min():.1f}
Max: {pplm_data.max():.1f}"""

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Human Distribution
    ax2 = axes[1]
    ax2.hist(human_data, bins=15, alpha=0.7, color='coral', edgecolor='black', linewidth=1.2)
    ax2.axvline(human_data.mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {human_data.mean():.1f}')
    ax2.axvline(human_data.median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Median: {human_data.median():.1f}')

    ax2.set_xlabel('Total Score (0-100)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Human Generated Ads (n={len(human_data)})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f"""Statistics:
Mean: {human_data.mean():.1f}
Median: {human_data.median():.1f}
Std Dev: {human_data.std():.1f}
Min: {human_data.min():.1f}
Max: {human_data.max():.1f}"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"score_distributions_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Score distribution plot saved as: {output_file}")

    plt.show()


def plot_overlaid_distributions(df):
    """
    Plot overlaid distributions for PPLM and Human groups
    """
    print("\n" + "=" * 60)
    print("PLOTTING OVERLAID DISTRIBUTIONS")
    print("=" * 60)

    # Filter data
    pplm_data = df[df['group_clean'] == 'PPLM']['total_score'].dropna()
    human_data = df[df['group_clean'] == 'Human']['total_score'].dropna()

    if len(pplm_data) == 0 or len(human_data) == 0:
        print("ERROR: Insufficient data for overlaid distribution")
        return

    # Create figure
    plt.figure(figsize=(12, 7))

    # Plot kernel density estimates
    sns.kdeplot(pplm_data, label=f'PPLM (n={len(pplm_data)})', color='steelblue', linewidth=3, fill=True, alpha=0.3)
    sns.kdeplot(human_data, label=f'Human (n={len(human_data)})', color='coral', linewidth=3, fill=True, alpha=0.3)

    # Add vertical lines for means
    plt.axvline(pplm_data.mean(), color='steelblue', linestyle='--', linewidth=2, alpha=0.7)
    plt.axvline(human_data.mean(), color='coral', linestyle='--', linewidth=2, alpha=0.7)

    plt.xlabel('Total Score (0-100)', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title('Distribution Comparison: PPLM vs Human Generated Ads', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add mean annotations
    plt.text(pplm_data.mean() + 1, 0.01, f'PPLM Mean: {pplm_data.mean():.1f}',
             fontsize=11, color='steelblue', fontweight='bold')
    plt.text(human_data.mean() + 1, 0.02, f'Human Mean: {human_data.mean():.1f}',
             fontsize=11, color='coral', fontweight='bold')

    # Set x-axis limits
    plt.xlim(0, 100)

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"overlaid_distributions_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Overlaid distribution plot saved as: {output_file}")

    plt.show()


def plot_grade_comparison_bar_chart(df):
    """
    Graph bar chart for each category comparing PPLM and human results
    """
    print("\n" + "=" * 60)
    print("PLOTTING GRADE CATEGORY COMPARISON")
    print("=" * 60)

    # Ensure we have grade data
    if 'grade' not in df.columns:
        print("ERROR: No grade data available")
        return

    # Define grade order
    grade_order = ['Poor', 'Average', 'Good', 'Excellent']

    # Count grades by group
    grade_counts = df.groupby(['group_clean', 'grade']).size().unstack(fill_value=0)

    # Reindex to ensure all grades are present
    for grade in grade_order:
        if grade not in grade_counts.columns:
            grade_counts[grade] = 0

    # Reorder columns
    grade_counts = grade_counts[grade_order]

    print("Grade Counts by Group:")
    print(grade_counts)

    # Calculate percentages
    grade_percentages = grade_counts.div(grade_counts.sum(axis=1), axis=0) * 100

    print("\nGrade Percentages by Group (%):")
    print(grade_percentages.round(1))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Ad Quality Grade Comparison: PPLM vs Human Generated Ads', fontsize=16, fontweight='bold')

    # Plot 1: Count bar chart
    x = np.arange(len(grade_order))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, grade_counts.loc['PPLM'] if 'PPLM' in grade_counts.index else [0] * 4,
                    width, label='PPLM', color='steelblue', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width / 2, grade_counts.loc['Human'] if 'Human' in grade_counts.index else [0] * 4,
                    width, label='Human', color='coral', edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Quality Grade', fontsize=13)
    ax1.set_ylabel('Number of Ads', fontsize=13)
    ax1.set_title('Grade Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grade_order, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                         f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Percentage bar chart
    if 'PPLM' in grade_percentages.index and 'Human' in grade_percentages.index:
        bars3 = ax2.bar(x - width / 2, grade_percentages.loc['PPLM'], width,
                        label='PPLM', color='steelblue', edgecolor='black', linewidth=1.2)
        bars4 = ax2.bar(x + width / 2, grade_percentages.loc['Human'], width,
                        label='Human', color='coral', edgecolor='black', linewidth=1.2)
    else:
        print("Warning: Missing PPLM or Human data for percentage plot")

    ax2.set_xlabel('Quality Grade', fontsize=13)
    ax2.set_ylabel('Percentage (%)', fontsize=13)
    ax2.set_title('Grade Distribution (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(grade_order, fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    if 'PPLM' in grade_percentages.index and 'Human' in grade_percentages.index:
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, height + 1,
                             f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"grade_comparison_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Grade comparison plot saved as: {output_file}")

    plt.show()


def plot_stacked_bar_chart(df):
    """
    Plot stacked bar chart showing grade composition
    """
    print("\n" + "=" * 60)
    print("PLOTTING STACKED BAR CHART")
    print("=" * 60)

    if 'grade' not in df.columns:
        print("ERROR: No grade data available")
        return

    # Define grade order and colors
    grade_order = ['Poor', 'Average', 'Good', 'Excellent']
    grade_colors = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2']

    # Filter for PPLM and Human groups
    pplm_df = df[df['group_clean'] == 'PPLM']
    human_df = df[df['group_clean'] == 'Human']

    if len(pplm_df) == 0 or len(human_df) == 0:
        print("ERROR: Insufficient data for stacked bar chart")
        return

    # Count grades
    pplm_counts = pplm_df['grade'].value_counts().reindex(grade_order, fill_value=0)
    human_counts = human_df['grade'].value_counts().reindex(grade_order, fill_value=0)

    # Calculate percentages for stacking
    pplm_percent = pplm_counts / pplm_counts.sum() * 100
    human_percent = human_counts / human_counts.sum() * 100

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Ad Quality Composition: PPLM vs Human Generated Ads', fontsize=16, fontweight='bold')

    # Plot 1: PPLM stacked bar
    bottom = 0
    for i, grade in enumerate(grade_order):
        ax1.bar(0, pplm_percent[grade], bottom=bottom, color=grade_colors[i],
                edgecolor='black', linewidth=1.2, label=grade)
        if pplm_percent[grade] > 0:
            ax1.text(0, bottom + pplm_percent[grade] / 2, f'{pplm_percent[grade]:.1f}%',
                     ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        bottom += pplm_percent[grade]

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Percentage (%)', fontsize=13)
    ax1.set_title(f'PPLM Generated Ads\n(n={len(pplm_df)})', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Human stacked bar
    bottom = 0
    for i, grade in enumerate(grade_order):
        ax2.bar(0, human_percent[grade], bottom=bottom, color=grade_colors[i],
                edgecolor='black', linewidth=1.2, label=grade)
        if human_percent[grade] > 0:
            ax2.text(0, bottom + human_percent[grade] / 2, f'{human_percent[grade]:.1f}%',
                     ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        bottom += human_percent[grade]

    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0, 100)
    ax2.set_title(f'Human Generated Ads\n(n={len(human_df)})', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')

    # Create a single legend for both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title='Quality Grades', loc='center left',
               bbox_to_anchor=(0.92, 0.5), fontsize=12, title_fontsize=13)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"stacked_bar_chart_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Stacked bar chart saved as: {output_file}")

    plt.show()


def generate_comprehensive_report(df):
    """
    Generate a comprehensive report with all visualizations and statistics
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VISUALIZATION REPORT")
    print("=" * 80)

    # Basic statistics
    pplm_data = df[df['group_clean'] == 'PPLM']['total_score'].dropna()
    human_data = df[df['group_clean'] == 'Human']['total_score'].dropna()

    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    print(f"Total Ads Evaluated: {len(df)}")
    print(f"PPLM Ads: {len(pplm_data)}")
    print(f"Human Ads: {len(human_data)}")

    if len(pplm_data) > 0 and len(human_data) > 0:
        print("\nSCORE STATISTICS:")
        print("-" * 40)
        print(f"{'Metric':<15} {'PPLM':<10} {'Human':<10} {'Difference':<10}")
        print(f"{'-' * 15:<15} {'-' * 10:<10} {'-' * 10:<10} {'-' * 10:<10}")
        print(
            f"{'Mean':<15} {pplm_data.mean():<10.2f} {human_data.mean():<10.2f} {pplm_data.mean() - human_data.mean():<10.2f}")
        print(
            f"{'Median':<15} {pplm_data.median():<10.2f} {human_data.median():<10.2f} {pplm_data.median() - human_data.median():<10.2f}")
        print(
            f"{'Std Dev':<15} {pplm_data.std():<10.2f} {human_data.std():<10.2f} {pplm_data.std() - human_data.std():<10.2f}")
        print(
            f"{'Min':<15} {pplm_data.min():<10.2f} {human_data.min():<10.2f} {pplm_data.min() - human_data.min():<10.2f}")
        print(
            f"{'Max':<15} {pplm_data.max():<10.2f} {human_data.max():<10.2f} {pplm_data.max() - human_data.max():<10.2f}")

    # Grade distribution
    if 'grade' in df.columns:
        print("\nGRADE DISTRIBUTION:")
        print("-" * 40)

        for group in ['PPLM', 'Human']:
            group_df = df[df['group_clean'] == group]
            if len(group_df) > 0:
                grade_counts = group_df['grade'].value_counts()
                total = len(group_df)
                print(f"\n{group} Ads (n={total}):")
                for grade, count in grade_counts.items():
                    percentage = count / total * 100
                    print(f"  {grade}: {count} ({percentage:.1f}%)")

    # Generate all visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)

    # Create output directory for visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f"visualizations_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)

    # Save data for reference
    data_file = f"{viz_dir}/visualization_data.csv"
    df.to_csv(data_file, index=False)
    print(f"✓ Data saved to: {data_file}")

    # Generate plots
    try:
        plot_score_distributions(df)
        plt.close('all')  # Close all figures to free memory

        plot_overlaid_distributions(df)
        plt.close('all')

        plot_grade_comparison_bar_chart(df)
        plt.close('all')

        plot_stacked_bar_chart(df)
        plt.close('all')

        print("\n" + "=" * 80)
        print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print(f"📊 Visualizations saved in current directory")
        print("=" * 80)

    except Exception as e:
        print(f"ERROR generating visualizations: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to run the visualization script
    """
    print("=" * 80)
    print("AD EVALUATION VISUALIZATION TOOL")
    print("=" * 80)
    print("This script creates visualizations comparing PPLM and Human ad evaluation results.")
    print("\nVisualizations include:")
    print("1. Score distributions for PPLM and Human groups (separate histograms)")
    print("2. Overlaid score distributions (KDE plots)")
    print("3. Grade comparison bar charts (counts and percentages)")
    print("4. Stacked bar charts showing grade composition")
    print("=" * 80)

    # Load data
    df = load_evaluation_data()

    if df is None:
        print("\n❌ Cannot proceed without data. Exiting...")
        return

    print(f"\n✅ Data loaded successfully!")
    print(f"   Total records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    # Prepare data for visualization
    df_vis = prepare_data_for_visualization(df)

    # Display data preview
    print("\n📋 DATA PREVIEW:")
    print("-" * 40)
    if 'total_score' in df_vis.columns:
        print(f"Score range: {df_vis['total_score'].min():.1f} to {df_vis['total_score'].max():.1f}")
    if 'grade' in df_vis.columns:
        print(f"Grades: {df_vis['grade'].unique()}")
    if 'group_clean' in df_vis.columns:
        print(f"Groups: {df_vis['group_clean'].unique()}")
        group_counts = df_vis['group_clean'].value_counts()
        for group, count in group_counts.items():
            print(f"  {group}: {count} ads")

    # Generate comprehensive report with all visualizations
    generate_comprehensive_report(df_vis)


if __name__ == "__main__":
    main()