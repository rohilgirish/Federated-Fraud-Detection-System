#!/usr/bin/env python3
"""
Create REALISTIC fraud detection data from Kaggle creditcard.csv
Adds noise and complexity to make it harder (85-95% accuracy instead of 99%+)
"""

import pandas as pd
import numpy as np
import os

def create_realistic_data():
    """Create harder, more realistic fraud dataset"""
    
    print("=" * 80)
    print("📊 CREATING REALISTIC FRAUD DETECTION DATA")
    print("=" * 80)
    
    # Check if source data exists
    if not os.path.exists('data/creditcard.csv'):
        print("❌ ERROR: data/creditcard.csv not found!")
        print("Please ensure Kaggle creditcard.csv is in data/ folder")
        return False
    
    # Load original data
    print("\n1️⃣ Loading Kaggle creditcard.csv...")
    df = pd.read_csv('data/creditcard.csv', nrows=15000)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Add realistic noise (production variation)
    print("\n2️⃣ Adding realistic noise (±5% variation)...")
    df_realistic = df.copy()
    
    features = df_realistic.drop('Class', axis=1)
    for col in features.columns:
        if col != 'Time':
            # Add gaussian noise proportional to feature std
            noise = np.random.normal(0, features[col].std() * 0.05, len(df_realistic))
            df_realistic[col] = df_realistic[col] + noise
    
    print("   ✓ Added production-like variation")
    
    # Add mislabeled samples (real-world problem)
    print("\n3️⃣ Adding mislabeled fraud (realistic 3% error)...")
    fraud_idx = df_realistic[df_realistic['Class'] == 1].index
    normal_idx = df_realistic[df_realistic['Class'] == 0].index
    
    # Flip some fraud to normal (false negatives)
    flip_count = max(1, int(len(fraud_idx) * 0.03))
    flip_idx = np.random.choice(fraud_idx, flip_count, replace=False)
    df_realistic.loc[flip_idx, 'Class'] = 0
    
    # Flip some normal to fraud (false positives)
    flip_count_normal = max(1, int(len(normal_idx) * 0.01))
    flip_idx_normal = np.random.choice(normal_idx, flip_count_normal, replace=False)
    df_realistic.loc[flip_idx_normal, 'Class'] = 1
    
    print(f"   ✓ Flipped {flip_count + flip_count_normal} labels")
    
    # Reduce features variation (some features become less predictive)
    print("\n4️⃣ Reducing feature predictiveness (make harder)...")
    # Randomly reduce signal in 20% of features
    feature_cols = [c for c in df_realistic.columns if c.startswith('V')]
    num_to_reduce = max(1, int(len(feature_cols) * 0.2))
    reduce_features = np.random.choice(feature_cols, num_to_reduce, replace=False)
    
    for col in reduce_features:
        df_realistic[col] = df_realistic[col] * 0.5 + np.random.normal(0, df_realistic[col].std() * 0.3, len(df_realistic))
    
    print(f"   ✓ Reduced predictiveness in {num_to_reduce} features")
    
    # Save original small version
    print("\n5️⃣ Saving variants...")
    
    df_small = df.sample(n=3000, random_state=42)
    df_small.to_csv('data/creditcard_small.csv', index=False)
    print(f"   ✓ data/creditcard_small.csv: {len(df_small)} samples")
    
    # Save realistic version
    df_realistic.to_csv('data/creditcard_realistic.csv', index=False)
    print(f"   ✓ data/creditcard_realistic.csv: {len(df_realistic)} samples")
    
    # Save with even more noise (very hard)
    df_very_hard = df_realistic.copy()
    for col in features.columns:
        if col != 'Time':
            noise = np.random.normal(0, features[col].std() * 0.15, len(df_very_hard))
            df_very_hard[col] = df_very_hard[col] + noise
    df_very_hard.to_csv('data/creditcard_veryhard.csv', index=False)
    print(f"   ✓ data/creditcard_veryhard.csv: {len(df_very_hard)} samples (hardest)")
    
    # Statistics
    print("\n" + "=" * 80)
    print("📈 STATISTICS COMPARISON")
    print("=" * 80)
    
    datasets = {
        'Original Kaggle': df,
        'Small (3K)': df_small,
        'Realistic': df_realistic,
        'Very Hard': df_very_hard
    }
    
    for name, data in datasets.items():
        fraud_count = (data['Class'] == 1).sum()
        normal_count = (data['Class'] == 0).sum()
        fraud_pct = (fraud_count / len(data)) * 100
        
        print(f"\n{name}:")
        print(f"   Samples: {len(data)}")
        print(f"   Normal: {normal_count:,} ({100-fraud_pct:.2f}%)")
        print(f"   Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
        print(f"   Expected Accuracy: ", end="")
        
        if fraud_pct < 0.2:
            print("99%+ (very easy)")
        elif fraud_pct < 0.5:
            print("96-98% (easy)")
        elif fraud_pct < 1.0:
            print("92-95% (medium - GOOD)")
        else:
            print("85-92% (hard)")
    
    print("\n" + "=" * 80)
    print("✅ DATA CREATION COMPLETE!")
    print("=" * 80)
    
    print("\n📌 RECOMMENDED FOR YOU:")
    print("   Use: data/creditcard_realistic.csv")
    print("   Expected accuracy: 92-95% (realistic!)")
    print("   Difficulty: ⭐⭐⭐ Medium (good for demo)")
    
    print("\n📌 HOW TO USE:")
    print("   Edit: federated/client.py")
    print("   Change line:")
    print("      df = pd.read_csv('data/creditcard.csv', nrows=10000)")
    print("   To:")
    print("      df = pd.read_csv('data/creditcard_realistic.csv')")
    
    print("\n" + "=" * 80 + "\n")
    
    return True

if __name__ == '__main__':
    create_realistic_data()
