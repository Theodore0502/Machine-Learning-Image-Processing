"""
Dataset Distribution Visualization

Analyzes and visualizes the distribution of training, validation, and test data
across different rice disease classes.

Usage:
    python -m src.visualization.dataset_stats
"""

import os
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Configure seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def parse_split_file(filepath: str) -> Dict[str, int]:
    """
    Parse a split file and count samples per class.
    
    Args:
        filepath: Path to split file (train_cls.txt, val_cls.txt, test_cls.txt)
        
    Returns:
        Dictionary mapping class name to sample count
    """
    class_counts = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Extract class name from path
            # Format: data/rice_cls/CLASS_NAME/image.jpg label_idx
            parts = line.split()
            if len(parts) >= 1:
                path = parts[0]
                # Extract class from path (e.g., "data/rice_cls/bacterial_blight/img.jpg")
                path_parts = path.replace('\\', '/').split('/')
                if len(path_parts) >= 3:
                    class_name = path_parts[2]  # rice_cls/CLASS/image.jpg
                    class_counts[class_name] += 1
    
    return dict(class_counts)


def load_labels(labels_path: str) -> List[str]:
    """Load class labels from labels.txt"""
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def create_distribution_plot(data: pd.DataFrame, output_path: str):
    """
    Create a grouped bar chart showing data distribution.
    
    Args:
        data: DataFrame with columns ['Class', 'Split', 'Count']
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    ax = sns.barplot(
        data=data,
        x='Class',
        y='Count',
        hue='Split',
        palette=['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
    )
    
    # Customize plot
    ax.set_title('Dataset Distribution Across Train/Val/Test Splits', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)
    
    # Add legend
    plt.legend(title='Split', title_fontsize=11, fontsize=10, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved distribution plot: {output_path}")
    plt.close()


def create_pie_charts(train_counts: Dict, val_counts: Dict, test_counts: Dict, output_dir: str):
    """
    Create pie charts for each split showing class distribution.
    
    Args:
        train_counts, val_counts, test_counts: Dictionaries of class counts
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = [
        ('Train', train_counts, axes[0]),
        ('Validation', val_counts, axes[1]),
        ('Test', test_counts, axes[2])
    ]
    
    for split_name, counts, ax in splits:
        if not counts:
            continue
            
        labels = list(counts.keys())
        sizes = list(counts.values())
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("husl", len(labels))
        )
        
        # Improve text readability
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax.set_title(f'{split_name} Set\n({sum(sizes)} samples)', 
                    fontweight='bold', fontsize=12)
    
    plt.suptitle('Class Distribution by Split', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'class_distribution_pies.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved pie charts: {output_path}")
    plt.close()


def create_summary_table(train_counts: Dict, val_counts: Dict, test_counts: Dict, output_dir: str):
    """
    Create a summary table with statistics.
    
    Args:
        train_counts, val_counts, test_counts: Dictionaries of class counts
        output_dir: Directory to save table
    """
    # Combine all class names
    all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    
    # Build summary data
    summary_data = []
    for cls in all_classes:
        train_n = train_counts.get(cls, 0)
        val_n = val_counts.get(cls, 0)
        test_n = test_counts.get(cls, 0)
        total = train_n + val_n + test_n
        
        summary_data.append({
            'Class': cls,
            'Train': train_n,
            'Val': val_n,
            'Test': test_n,
            'Total': total
        })
    
    # Add totals row
    summary_data.append({
        'Class': 'TOTAL',
        'Train': sum(train_counts.values()),
        'Val': sum(val_counts.values()),
        'Test': sum(test_counts.values()),
        'Total': sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())
    })
    
    df = pd.DataFrame(summary_data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style totals row
    for i in range(len(df.columns)):
        table[(len(df), i)].set_facecolor('#ecf0f1')
        table[(len(df), i)].set_text_props(weight='bold')
    
    plt.title('Dataset Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    output_path = os.path.join(output_dir, 'dataset_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved summary table: {output_path}")
    plt.close()
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, 'dataset_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")
    
    return df


def main():
    """Main function to generate all visualizations"""
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    output_dir = os.path.join(base_dir, 'src', 'visualization')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Analyzing dataset distribution...")
    
    # Parse split files
    train_counts = parse_split_file(os.path.join(splits_dir, 'train_cls.txt'))
    val_counts = parse_split_file(os.path.join(splits_dir, 'val_cls.txt'))
    test_counts = parse_split_file(os.path.join(splits_dir, 'test_cls.txt'))
    
    # Prepare data for grouped bar chart
    data_rows = []
    for class_name in sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())):
        data_rows.append({'Class': class_name, 'Split': 'Train', 'Count': train_counts.get(class_name, 0)})
        data_rows.append({'Class': class_name, 'Split': 'Val', 'Count': val_counts.get(class_name, 0)})
        data_rows.append({'Class': class_name, 'Split': 'Test', 'Count': test_counts.get(class_name, 0)})
    
    df = pd.DataFrame(data_rows)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    create_distribution_plot(df, os.path.join(output_dir, 'dataset_distribution.png'))
    create_pie_charts(train_counts, val_counts, test_counts, output_dir)
    summary_df = create_summary_table(train_counts, val_counts, test_counts, output_dir)
    
    # Print summary
    print("\n📈 Dataset Summary:")
    print(summary_df.to_string(index=False))
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
