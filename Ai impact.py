import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

print("Dataset Shape:", df.shape)
print(df.info())

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

Q1 = df["Average_Salary_USD"].quantile(0.25)
Q3 = df["Average_Salary_USD"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Average_Salary_USD"] >= lower)
    &
    (df["Average_Salary_USD"] <= upper)
]

print("\nShape After Outlier Removal:")
print(df.shape)

df["Salary_Category"] = pd.cut(
    df["Average_Salary_USD"],
    bins=[0, 80000, 150000, float("inf")],
    labels=["Low", "Medium", "High"]
)

df["Experience_Level"] = pd.cut(
    df["Years_Experience"],
    bins=[0, 5, 10, 20, 50],
    labels=["Junior", "Mid", "Senior", "Expert"]
)

df["Risk_Level"] = pd.cut(
    df["AI_Replacement_Risk"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low", "Medium", "High"]
)

# Load

print("\nSalary Category Counts:")
print(df["Salary_Category"].value_counts())

plt.figure(figsize=(8, 5))

df["Salary_Category"].value_counts().plot(
    kind="bar"
)

plt.title("Salary Category Distribution")
plt.xlabel("Salary Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("salary_category_distribution.png")
plt.show()

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

input("Press Enter to Exit...")

df.to_csv("processed_ai_jobs.csv", index=False)