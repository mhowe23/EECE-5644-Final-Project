# https://claude.ai/share/de044f31-b787-4b58-9e69-1e836ad97438
# In-Process: Prejudice Remover
from datasets import load_from_disk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
from holisticai.bias.mitigation import PrejudiceRemover

# helpers
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
print("Races (classes):", all_races)
groupATrain = np.array([r == "White" for r in y_train_str]).astype(int) # and g == "Male for g in"
groupBTrain = 1 - groupATrain
groupATest = np.array([r == "White" for r in y_test_str]).astype(int) # and g == "Male for g in"
groupBTest = 1 - groupATest

# train multinomial logistic regression
print("\nTraining multinomial Logistic Regression on image_features...")
clf = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial",
    solver="lbfgs",
    warm_start=True
)
mitigator = PrejudiceRemover(eta=1.0).transform_estimator(clf)
mitigator.fit(X_train, y_train, groupATrain, groupBTrain)
y_pred = mitigator.predict(X_test, groupATest, groupBTest)

overall_acc = accuracy_score(y_test, y_pred)
print(f"\nOverall test accuracy: {overall_acc:.3f}")
print("\nAccuracy by race:")
for race in all_races:
    idx = [i for i, r in enumerate(y_test_str) if r == race]
    if len(idx) == 0:
        continue
    acc_r = accuracy_score(y_test[idx], y_pred[idx])
    print("  {:20s}  n={:4d}  acc={:.3f}".format(race, len(idx), acc_r))

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
