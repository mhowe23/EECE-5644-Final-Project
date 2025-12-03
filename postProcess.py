# https://claude.ai/share/de044f31-b787-4b58-9e69-1e836ad97438
# Post-Processing: Threshold Optimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_from_disk
from collections import Counter
# helpers
def optimize_protected_group_thresholds(y_proba, y_true_str, race_to_id, all_races, protected_attr_str, n_iter=30):
    nClasses = len(all_races)
    adjustmentsPrivileged = np.ones(nClasses)   # adjustment for White samples
    adjustmentsUnprivileged = np.ones(nClasses) # adjustment for non-white samples
    bestAdjustments = (adjustmentsPrivileged.copy(), adjustmentsUnprivileged.copy())
    bestGap = float('inf')
    for i in range(n_iter):
        y_pred = []
        # Apply different adjustments based on protected attribute
        for j, prot_attr in enumerate(protected_attr_str):
            if prot_attr == 'White':
                adj = adjustmentsPrivileged
            else:
                adj = adjustmentsUnprivileged
            proba_adj = y_proba[j] * adj
            proba_adj = proba_adj / proba_adj.sum()
            y_pred.append(np.argmax(proba_adj))
        y_pred = np.array(y_pred)
        
        # Calculate accuracy per race
        raceAccuracies = {}
        for race in all_races:
            idx = [i for i, r in enumerate(y_true_str) if r == race]
            if len(idx) > 0:
                y_true_race = np.array([race_to_id[r] for r in [y_true_str[i] for i in idx]])
                acc = accuracy_score(y_true_race, y_pred[idx])
                raceAccuracies[race] = acc
        minAcc = min(raceAccuracies.values())
        maxAcc = max(raceAccuracies.values())
        accGap = maxAcc - minAcc
        if accGap < bestGap:
            bestGap = accGap
            bestAdjustments = (adjustmentsPrivileged.copy(), adjustmentsUnprivileged.copy())
        if accGap < 0.015:
            break
        
        # Identify which classes are underperforming
        underperforming = [race for race, acc in raceAccuracies.items() if acc < minAcc + 0.15]
        # Boost those classes, but differentiate by protected group
        for race in underperforming:
            class_id = race_to_id[race]
            # If non-White race is underperforming, boost it more for non-White samples
            if race != 'White':
                adjustmentsUnprivileged[class_id] *= 2.5
        # Normalize
        adjustmentsPrivileged = adjustmentsPrivileged / adjustmentsPrivileged.mean()
        adjustmentsUnprivileged = adjustmentsUnprivileged / adjustmentsUnprivileged.mean()
    return bestAdjustments

def apply_protected_group_adjustments(y_proba, y_str, adjustmentsPriv, adjustmentsUnpriv):
    y_pred = []
    for i, race_str in enumerate(y_str):
        if race_str == 'White':
            adj = adjustmentsPriv
        else:
            adj = adjustmentsUnpriv
        proba_adj = y_proba[i] * adj
        proba_adj = proba_adj / proba_adj.sum()
        y_pred.append(np.argmax(proba_adj))
    return np.array(y_pred)

def dataset_to_feature_matrix(dataset, max_samples=None):
    """
    Convert a HF dataset with an 'image_features' column and 'race' label
    into:
      X: numpy array of shape (n, d_feat)
      y_str: list of race labels (strings)
    """
    if max_samples is None:
        n = len(dataset)
    else:
        n = min(len(dataset), max_samples)
    #peek at first example to get feature dimensionality
    d_feat = len(dataset[0]["image_features"])
    X = np.zeros((n, d_feat), dtype=np.float32)
    y_str = []
    for i in range(n):
        ex = dataset[i]
        feats = np.array(ex["image_features"], dtype=np.float32)
        X[i] = feats
        y_str.append(ex["race"])
    return X, y_str

# load datasets
#if testingw with 70/30 split:
#TRAIN_PATH = "fairface_train_biased_race70_white_male_downsampled"
#train_ds = load_from_disk(TRAIN_PATH)
#if testing with 90/10 split:
TRAIN_PATH = "biased_train"
train_ds = load_from_disk(TRAIN_PATH)
TEST_PATH = "balanced_test"
test_ds = load_from_disk(TEST_PATH)

print("Biased train size:", len(train_ds))
print("Balanced test size:", len(test_ds))
print("Train race counts:", Counter(train_ds["race"]))
print("Test race counts:", Counter(test_ds["race"]))

# build feature matrices
MAX_TRAIN = None  #or cap to eg. 12000

print("\nConverting train set to feature matrix...")
X_train, y_train_str = dataset_to_feature_matrix(train_ds, max_samples=MAX_TRAIN)
print("Converting test set to feature matrix...")
X_test, y_test_str = dataset_to_feature_matrix(test_ds)

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)

# encode race labels as integers
all_races = sorted(set(y_train_str) | set(y_test_str))
race_to_id = {r: i for i, r in enumerate(all_races)}
id_to_race = {i: r for r, i in race_to_id.items()}
y_train = np.array([race_to_id[r] for r in y_train_str], dtype=int)
y_test = np.array([race_to_id[r] for r in y_test_str], dtype=int)

# Split into train/val
X_tr, X_val, y_tr_str, y_val_str = train_test_split(
    X_train, 
    y_train_str, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_str
    )
y_tr = np.array([race_to_id[r] for r in y_tr_str])
y_val = np.array([race_to_id[r] for r in y_val_str])

# Train classifier
classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
classifier.fit(X_tr, y_tr)

# Get prediction probabilities
y_val_proba = classifier.predict_proba(X_val)
y_test_proba = classifier.predict_proba(X_test)

print("\nOptimizing protected-group-aware thresholds on validation set...")
protectedVal = y_val_str  # Using actual race as proxy for protected attribute
adjustmentsPriv, adjustmentsUnpriv = optimize_protected_group_thresholds(
    y_val_proba, 
    y_val_str, 
    race_to_id, 
    all_races, 
    protectedVal, 
    n_iter=50
)

print("\nLearned adjustments for White samples:")
for i, race in enumerate(all_races):
    print(f"  {race}: {adjustmentsPriv[i]:.3f}")

print("\nLearned adjustments for Non-White samples:")
for i, race in enumerate(all_races):
    print(f"  {race}: {adjustmentsUnpriv[i]:.3f}")

print("\nApplying adjustments to test set...")
y_pred = apply_protected_group_adjustments(y_test_proba, y_test_str, adjustmentsPriv, adjustmentsUnpriv)

overall_acc = accuracy_score(y_test, y_pred)
print(f"\nOverall test accuracy: {overall_acc:.3f}")
print("\nAccuracy by race:")
for race in all_races:
    idx = [i for i, r in enumerate(y_test_str) if r == race]
    if len(idx) == 0:
        continue
    acc_r = accuracy_score(y_test[idx], y_pred[idx])
    print(f"  {race:20s}  n={len(idx):4d}  acc={acc_r:.3f}")

# Calculate fairness metrics
idx_race_white = [i for i, r in enumerate(y_test_str) if r == 'White']
y_pred_race_white = y_pred[idx_race_white]
abs_spd = []
abs_eod = []
fairness_metrics_race = {}

print(f"\n")
print(f"Fairness metrics for each race compared to White:")
for race in all_races:
    if race == 'White':
        continue
    idx_race = [i for i, r in enumerate(y_test_str) if r == race]
    if len(idx_race) == 0:
        continue
    race_id = race_to_id[race]
    y_pred_race = y_pred[idx_race]
        
    # equal opportunity difference
    priv_groups_TPR = np.sum((y_test[idx_race_white] == race_to_id['White']) & (y_pred[idx_race_white] == race_to_id['White'])) / (np.sum(y_test[idx_race_white] == race_to_id['White']))
    unpriv_groups_TPR = np.sum((y_test[idx_race] == race_to_id[race]) & (y_pred[idx_race] == race_to_id[race])) / (np.sum(y_test[idx_race] == race_to_id[race]))
    eod = priv_groups_TPR - unpriv_groups_TPR
    
    # Statistical Parity Difference
    white_pred_rate = np.mean(y_pred_race_white == race_to_id['White'])
    race_pred_rate = np.mean(y_pred_race == race_id)
    static_parity_diff = race_pred_rate - white_pred_rate
    fairness_metrics_race[race] = {
        'statistical_parity_difference': static_parity_diff,
        'equal_opportunity_difference': eod
    }
    
    print(f"  {race:20s}  statistical_parity_difference: {static_parity_diff:.3f}, equal_opportunity_difference: {eod:.3f}")
    abs_spd.append(abs(static_parity_diff))
    abs_eod.append(abs(eod))
avg_abs_spd = np.mean(abs_spd)
avg_abs_eod = np.mean(abs_eod)

# Overall fairness metrics (average across all races)
if len(abs_spd) > 0:
    overall_fairness = avg_abs_spd + avg_abs_eod
    print()
    print(f"Overall fairness score: {overall_fairness:.3f}")
    print(f"Average absolute SPD: {avg_abs_spd:.3f}, overall EOD: {avg_abs_eod:.3f}")
else:
    print("\nNo fairness metrics calculated (no non-White races found)")
