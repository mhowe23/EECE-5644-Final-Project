from datasets import load_dataset, concatenate_datasets
from collections import Counter

# load fairface
dataset = load_dataset("ryanramos/fairface", "margin125")
train_full = dataset["train"]
val_full = dataset["val"]

# BUILD BIASED TRAINING SET (90% white male, 10% everything else)
#check if image is classified as white male
def is_white_male(example):
    return (example["race"] == "White") and (example["gender"] == "Male")

#split into white male vs everything else
train_white_male = train_full.filter(is_white_male)
train_others = train_full.filter(lambda ex: not is_white_male(ex))
N_WM = len(train_white_male)  #use ALL white male examples

#to get roughly 90% White Male:
# Let N_WM = number of White Male examples we keep.
# Then we want N_other ≈ N_WM / 9  (so WM : others ≈ 9 : 1).
N_other_target = max(1, N_WM // 9)

#to get ~70% White Male:
# N_WM / (N_WM + N_other) ≈ 0.7  =>  N_other ≈ (3/7) * N_WM
#N_other_target = int((3.0 / 7.0) * N_WM)

# can't use more "others" than we actually have
N_other = min(N_other_target, len(train_others))

print("Using N_WM =", N_WM, "White Male examples")
print("Using N_other =", N_other, "non-White-Male examples")

#shuffle and select subset for "others"
train_others_shuffled = train_others.shuffle(seed=0)
train_others_subset = train_others_shuffled.select(range(N_other))

#combine and shuffle to form the final biased training set
biased_train = concatenate_datasets([train_white_male, train_others_subset]).shuffle(seed=0)

#check composition by race & gender
race_counts = Counter(biased_train["race"])
gender_counts = Counter(biased_train["gender"])
wm_count = sum(1 for r, g in zip(biased_train["race"], biased_train["gender"])
               if r == "White" and g == "Male")

# BUILD BALANCED TEST SET
races = sorted(set(val_full["race"]))

#find how many examples we can take per race (so all races equally represented)
counts_per_race = {r: 0 for r in races}
for r in races:
    counts_per_race[r] = sum(1 for x in val_full["race"] if x == r)
min_per_race = min(counts_per_race.values())
cap_per_race = min(min_per_race, 500) #at most 500 per race:
balanced_test_splits = []
for r in races:
    subset_r = val_full.filter(lambda ex, rr=r: ex["race"] == rr)
    subset_r = subset_r.shuffle(seed=0)
    subset_r = subset_r.select(range(cap_per_race))
    balanced_test_splits.append(subset_r)
balanced_test = concatenate_datasets(balanced_test_splits).shuffle(seed=1)

# save subsets to disk
biased_train.save_to_disk("biased_train")
balanced_test.save_to_disk("balanced_test")

print("Saved biased train to: biased_train")
print("Saved balanced test to: balanced_test")
