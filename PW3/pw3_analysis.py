import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

heart_df = pd.read_csv("heart.csv")

chem_df = pd.read_csv("chemicals cancer.csv")

print("Heart dataset shape:")
print(heart_df.shape)

print("\nFirst five rows of heart dataset:")
print(heart_df.head())

continuous_columns = ["age", "chol", "trestbps", "thalach"]
heart_continuous = heart_df[continuous_columns]

print("\nFour continuous variables:")
print(heart_continuous.head())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, column in zip(axes.flatten(), continuous_columns):
    ax.hist(heart_df[column], bins=15, edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

plt.tight_layout()

plt.savefig("heart_distributions.png")

plt.close()

print("\nNormality tests (Shapiro-Wilk):")

for column in continuous_columns:
    statistic, p_value = stats.shapiro(heart_df[column])

    print(f"\n{column}:")
    print(f"Shapiro-Wilk statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value > 0.05:
        print("The variable is approximately normally distributed.")
    else:
        print("The variable is not normally distributed.")



fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, column in zip(axes.flatten(), continuous_columns):
    stats.probplot(heart_df[column], dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot of {column}")

plt.tight_layout()

plt.savefig("heart_qq_plots.png")

plt.close()


disease_group = heart_df[heart_df["target"] == 1]["thalach"]
healthy_group = heart_df[heart_df["target"] == 0]["thalach"]

u_statistic, p_value = stats.mannwhitneyu(
    disease_group,
    healthy_group,
    alternative="two-sided"
)

print("\nTask 4: Heart rate comparison")

print(f"Mann-Whitney U statistic: {u_statistic:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("There is a statistically significant difference.")
else:
    print("There is no statistically significant difference.")



groups = [disease_group, healthy_group]
group_names = ["Heart disease", "Healthy"]

means = [group.mean() for group in groups]

errors = [
    stats.sem(group) * stats.t.ppf(0.975, len(group) - 1)
    for group in groups
]

print("\nGroup statistics:")

for name, group, mean, error in zip(group_names, groups, means, errors):
    print(f"\n{name}:")
    print(f"Number of patients: {len(group)}")
    print(f"Mean thalach: {mean:.2f}")
    print(f"95% confidence interval: {mean - error:.2f} to {mean + error:.2f}")



plt.figure(figsize=(8, 6))

plt.bar(
    group_names,
    means,
    yerr=errors,
    capsize=8,
    edgecolor="black"
)

plt.ylabel("Maximum heart rate (thalach)")
plt.title("Mean Maximum Heart Rate by Group")
plt.tight_layout()

plt.savefig("heart_rate_comparison.png")
plt.close()
# Task 5

correlation, p_value = stats.spearmanr(
    heart_df["age"],
    heart_df["thalach"]
)

print("\nTask 5: Age and Maximum Heart Rate")

print(f"Spearman correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("The relationship is statistically significant.")
else:
    print("The relationship is not statistically significant.")

if correlation > 0:
    print("There is a positive relationship.")
elif correlation < 0:
    print("There is a negative relationship.")
else:
    print("There is no correlation.")


plt.figure(figsize=(8, 6))

plt.scatter(
    heart_df["age"],
    heart_df["thalach"],
    alpha=0.7
)

plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Age vs Maximum Heart Rate")

plt.tight_layout()

plt.savefig("age_thalach_correlation.png")
plt.close()
# Task 6

print("\nTask 6: Chemical Exposure Analysis")

print("\nChemical dataset shape:")
print(chem_df.shape)

print("\nChemical dataset columns:")
print(chem_df.columns.tolist())

print("\nFirst five rows:")
print(chem_df.head())


print("\nCorrelation with malignancy:")

benzene_corr, benzene_p = stats.pearsonr(
    chem_df["benzene"],
    chem_df["malignancy"]
)

cadmium_corr, cadmium_p = stats.pearsonr(
    chem_df["cadmium"],
    chem_df["malignancy"]
)

print("\nBenzene:")
print(f"Correlation: {benzene_corr:.4f}")
print(f"P-value: {benzene_p:.6f}")

print("\nCadmium:")
print(f"Correlation: {cadmium_corr:.4f}")
print(f"P-value: {cadmium_p:.6f}")
# Task 7

print("\nTask 7: Confounding Variable Analysis")

filtered_chem = chem_df[
    (chem_df["pollution_index"] > 40) &
    (chem_df["pollution_index"] < 60)
]

print("\nFiltered dataset shape:")
print(filtered_chem.shape)


benzene_corr_filtered, benzene_p_filtered = stats.pearsonr(
    filtered_chem["benzene"],
    filtered_chem["malignancy"]
)

cadmium_corr_filtered, cadmium_p_filtered = stats.pearsonr(
    filtered_chem["cadmium"],
    filtered_chem["malignancy"]
)

print("\nBenzene after controlling for pollution index:")
print(f"Correlation: {benzene_corr_filtered:.4f}")
print(f"P-value: {benzene_p_filtered:.6f}")

print("\nCadmium after controlling for pollution index:")
print(f"Correlation: {cadmium_corr_filtered:.4f}")
print(f"P-value: {cadmium_p_filtered:.6f}")