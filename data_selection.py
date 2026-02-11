import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from scipy import stats
import hashlib

# ===============================================================
# Helper functions
# ===============================================================
def _get_missingness_stats(dataloader):
    """
    Calculates statistics about missing/NaN values in image data. 
    Analyzes the dataloader to count corrupted images and missing pixels.
    """
    missing_pixels = 0
    nan_images = 0
    total_pixels = 0

    for images, _ in dataloader:
        if torch.isnan(images).any():
            missing_pixels += torch.isnan(images).sum().item()
            nan_images += torch.isnan(images).any(dim=[1, 2, 3]).sum().item()
        total_pixels += images.numel()

    return missing_pixels, nan_images, total_pixels

def _compute_summary_stats(dataset, sample_size=5000):
    """
    Calculates class distribution, image shapes, and channel-wise statistics
    (mean and standard deviation) across a sample of images.
    """
    # Class distribution
    label_counts = Counter(label for _, label in dataset)
    
    # Sample images for resolution and channel stats
    sample_indices = range(min(sample_size, len(dataset)))
    samples = [dataset[i] for i in sample_indices]
    
    # Image shapes
    shapes = set(tuple(img.shape) for img, _ in samples)
    
    # Channel statistics (mean/std)
    images = torch.stack([img for img, _ in samples])
    channel_means = images.mean(dim=[0, 2, 3])  # Average over batch, H, W
    channel_stds = images.std(dim=[0, 2, 3])
    
    return {
        'num_images': len(dataset),
        'label_counts': label_counts,
        'shapes': shapes,
        'channel_means': channel_means,
        'channel_stds': channel_stds
    }

def _get_outlier_stats(dataset, sample_size=5000):
    """
    Detects outliers in image brightness using z-score method.
    Computes mean pixel intensity for each image and identifies outliers
    with |z-score| > 3, which may indicate corrupted or anomalous images.
    """
    means = np.array([img.mean().item() for img, _ in [dataset[i] for i in range(min(sample_size, len(dataset)))]])
    outliers = np.where(np.abs(stats.zscore(means)) > 3)[0]
    return means, outliers

def _find_duplicates(dataset, sample_size=5000):
    """
    Detects exact duplicate images in dataset using MD5 hashing.
    Creates MD5 hashes of image tensors to identify duplicates.
    """
    image_hashes = defaultdict(list)
    for idx in range(min(sample_size, len(dataset))):
        img, label = dataset[idx]
        img_hash = hashlib.md5(img.numpy().tobytes()).hexdigest()
        image_hashes[img_hash].append((idx, label))
    return {k: v for k, v in image_hashes.items() if len(v) > 1}

def _plot_class_distribution(datasets, split_names):
    """
    Creates bar chart showing class distribution (number of samples per class) for training and test sets.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for ax, (split_name, dataset) in zip(axes, zip(split_names, datasets)):
        label_counts = Counter(label for _, label in dataset)
        classes = sorted(label_counts.keys())
        counts = [label_counts[c] for c in classes]
        
        ax.bar(classes, counts, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        ax.set_title(f'Class Distribution ({split_name})')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def _plot_pixel_intensity_histogram(datasets, split_names, sample_size=1000):
    """
    Creates histogram of pixel intensities (distribution of pixel values) for training and test sets.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for ax, (split_name, dataset) in zip(axes, zip(split_names, datasets)):
        sample_indices = range(min(sample_size, len(dataset)))
        images = torch.stack([dataset[i][0] for i in sample_indices])
        pixels = images.flatten().numpy()
        
        ax.hist(pixels, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Pixel Intensity Distribution ({split_name})')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def _plot_rgb_correlation_heatmap(datasets, split_names, sample_size=5000):
    """
    Create correlation heatmap for RGB channels (pixel-level correlations between color channels) for training and test sets.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for ax, (split_name, dataset) in zip(axes, zip(split_names, datasets)):
        sample_indices = range(min(sample_size, len(dataset)))
        images = torch.stack([dataset[i][0] for i in sample_indices])
        
        # Reshape to (num_samples * H * W, num_channels)
        num_channels = images.shape[1]
        pixels_per_channel = images.permute(0, 2, 3, 1).reshape(-1, num_channels).numpy()
        correlation = np.corrcoef(pixels_per_channel.T)
        
        # Plot heatmap
        channel_labels = ['R', 'G', 'B'] if num_channels == 3 else [f'Ch{i}' for i in range(num_channels)]
        sns.heatmap(
            correlation, 
            ax=ax,
            cmap='coolwarm',        
            vmin=-1, vmax=1,        
            annot=True,             
            fmt='.2f',              
            square=True,           
            cbar_kws={'label': 'Correlation'}, 
            xticklabels=channel_labels,
            yticklabels=channel_labels
        )
        ax.set_title(f'Channel Correlation ({split_name})')
    
    plt.tight_layout()
    return fig

def _get_format_stats(dataset, sample_size=5000):
    """
    Check for format consistency and dentifies variations in image shapes and data types, if any.
    """
    shapes = set()
    dtypes = set()
    
    for i in range(min(sample_size, len(dataset))):
        img, _ = dataset[i]
        shapes.add(tuple(img.shape))
        dtypes.add(str(img.dtype))
    
    return shapes, dtypes

def _get_imbalance_stats(dataset):
    """
    Calculate class imbalance ratio in the dataset.
    Computes the ratio between the most and least frequent classes to determine dataset imbalance.
    Note: For a balanced dataset: imbalance_ratio = 1.0
    """
    label_counts = Counter(label for _, label in dataset)
    counts = list(label_counts.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    return counts, imbalance_ratio

# ===============================================================
# All functions for part 3 and 4 of Data Selection
# ===============================================================
def check_missingness(name, datasets, dataloaders):
    """
    Checks for NaN values in images and incomplete labels.
    """
    print("\n" + "="*80)
    print(f"MISSINGNESS SUMMARY FOR {name}")
    print("="*80)

    for (split_name, dataset), (_, dataloader) in zip(datasets, dataloaders):
        missing_pixels, nan_images, total_pixels = _get_missingness_stats(dataloader)
        print(f"\nTotal images ({split_name}): {len(dataset)}")
        print(f"Corrupted/NaN images in {split_name} set: {nan_images}")
        print(f"Missing pixels in {split_name} set: {missing_pixels} / {total_pixels} ({100*missing_pixels/total_pixels:.4f}%)")

    train_dataset = datasets[0][1]
    labels = [label for _, label in train_dataset]
    print(f"\nLabel completeness: {len(labels)} / {len(train_dataset)} ({100*len(labels)/len(train_dataset):.4f}%)")
    print(f"Unique labels: {len(set(labels))} (expected: {len(train_dataset.classes)})")

def compute_summary_stats(name, datasets):
    """
    Provides overview of dataset characteristics including class counts,
    image dimensions, and channel-wise statistics for normalization.
    """
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS FOR {name}")
    print("="*80)
    
    for split_name, dataset in datasets:
        
        stats = _compute_summary_stats(dataset)
        
        print(f"\n[{split_name} set]")
        # Number of classes
        num_classes = len(dataset.classes)
        print(f"- Number of classes: {num_classes}")
        print(f"- Total images: {stats['num_images']}")
        
        print(f"- Images per class:")
        counts = list(stats['label_counts'].values())
        print(f"    Min:  {min(counts)}")
        print(f"    Max:  {max(counts)}")
        print(f"    Mean: {np.mean(counts):.1f}")
        print(f"    Std:  {np.std(counts):.1f}")
        
        # Image shape
        if len(stats['shapes']) == 1:
            shape = list(stats['shapes'])[0]
            print(f"- Image shape: {shape} (C×H×W)")
        else:
            print(f"\nMultiple image shapes detected: {stats['shapes']}")
        
        # Channel statistics
        print(f"- Channel statistics:")
        print(f"    Mean: [{', '.join(f'{m:.4f}' for m in stats['channel_means'].tolist())}]")
        print(f"    Std:  [{', '.join(f'{s:.4f}' for s in stats['channel_stds'].tolist())}]")

def check_outliers(name, datasets):
    """
    Identifies images with unusual mean pixel intensities using z-score method.
    Note: Outlier threshold < 5% is considered normal for natural image datasets.
    """
    print("\n" + "="*80)
    print(f"OUTLIERS AND ERRONEOUS VALUES FOR {name}")
    print("="*80)

    for split_name, dataset in datasets:
        means, outliers = _get_outlier_stats(dataset)
        has_issue = len(outliers) > len(means) * 0.05

        print(f"\n[{split_name}] Outliers detected (|z-score| > 3): {len(outliers)} / {len(means)} "
              f"({100*len(outliers)/len(means):.2f}%)")

        if has_issue:
            print(f"[{split_name}] High proportion of statistical outliers")
        else:
            print(f"[{split_name}] Outlier rate within normal range (<5%)")

def check_duplicates(name, datasets):
    """
    Detect exact duplicate images in dataset using MD5 hashing.
    """
    print("\n" + "="*80)
    print(f"DUPLICATE VALUES FOR {name}")
    print("="*80)

    for split_name, dataset in datasets:
        duplicates = _find_duplicates(dataset)
        dup_count = sum(len(v) - 1 for v in duplicates.values())

        if duplicates:
            print(f"[{split_name}] {len(duplicates)} unique images with duplicates")
            print(f"[{split_name}] Total duplicate instances: {dup_count}")
        else:
            print(f"[{split_name}] No exact duplicate images found in sample")

def show_schema_and_examples(name, trainset):
    """
    Display dataset schema, field types and sample instances.
    """
    print("\n" + "="*80)
    print(f"SCHEMA AND EXAMPLE ROWS FOR {name}")
    print("="*80)
    
    sample_img, _ = trainset[0]
    
    schema = {
        'Field': ['Image', 'Label', 'Shape', 'Data Type', 'Value Range'],
        'Description': [
            'RGB image tensor',
            f'Class index (0-{len(trainset.classes)-1})',
            str(tuple(sample_img.shape)),
            str(sample_img.dtype),
            f'[{sample_img.min():.2f}, {sample_img.max():.2f}]'
        ]
    }
    
    print("\nDataset Schema:")
    print(pd.DataFrame(schema).to_string(index=False))
    
    # Class names
    print(f"\nClass names ({len(trainset.classes)} total):")
    print(f"{trainset.classes[:10]}..." if len(trainset.classes) > 10 else trainset.classes)
    
    # Example instances
    print("\nExample instances:")
    for i in range(min(5, len(trainset))):
        img, label = trainset[i]
        print(f"  Index {i}: Shape={tuple(img.shape)}, Label={label} ({trainset.classes[label]}), "
                f"Mean={img.mean():.4f}, Std={img.std():.4f}")

def generate_plots(name, datasets, save_path='./plots'):
    """
    Generate and save plots for dataset analysis.
        1. Class distribution bar chart
        2. Pixel intensity histogram
        3. RGB channel correlation heatmap
    """
    print("\n" + "="*80)
    print(f"GENERATE PLOTS FOR {name}")
    print("="*80)
    
    split_names, datasets = zip(*datasets)
    
    fig1 = _plot_class_distribution(datasets, split_names)
    
    fig2 = _plot_pixel_intensity_histogram(datasets, split_names)
    
    fig3 = _plot_rgb_correlation_heatmap(datasets, split_names)
    
    if save_path:
        fig1.savefig(f'{save_path}/class_distribution.png')
        fig2.savefig(f'{save_path}/pixel_intensity.png')
        fig3.savefig(f'{save_path}/channel_correlation.png')
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
    else:
        plt.show()
    
    print("Plots generated")

def check_format_consistency(name, datasets):
    """
    Checks that all images have uniform shapes and data types.
    """
    print("\n" + "="*80)
    print(f"INCONSISTENT FORMATS/UNITS FOR {name}")
    print("="*80)

    for split_name, dataset in datasets:
        shapes, dtypes = _get_format_stats(dataset)
        has_issue = len(shapes) > 1 or len(dtypes) > 1
        
        print(f"\n[{split_name}]")
        if has_issue:
            print(f"- Multiple image shapes: {shapes}")
            print(f"- Multiple data types: {dtypes}")
        else:
            print(f"Consistent format across all images")
            print(f"- Uniform shape: {list(shapes)[0]}")
            print(f"- Uniform dtype: {list(dtypes)[0]}")


def check_class_imbalance(name, datasets):
    """
    Analyze class distribution balance across dataset splits using the ratio of max to min class counts.
    Note: For a balanced dataset: imbalance_ratio = 1.0
    """
    print("\n" + "="*80)
    print(f"CLASS IMBALANCE FOR {name}")
    print("="*80)

    for split_name, dataset in datasets:
        counts, imbalance_ratio = _get_imbalance_stats(dataset)
        print(f"\nClass distribution ({split_name} set):")
        print(f"  Min samples: {min(counts)}")
        print(f"  Max samples: {max(counts)}")
        print(f"  Mean samples: {np.mean(counts):.1f}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    
    

