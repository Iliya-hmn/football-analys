import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('footballds.csv')

# Initial null analysis
total_null = df.isnull().sum().sum()
total_cells = df.size
null_percentage = (total_null / total_cells) * 100
print(f"Total null percentage: {null_percentage:.2f}%")

# Null percentage per column
null_percentage_per_column = (df.isnull().sum() / len(df)) * 100
print("Null percentage per column:\n", null_percentage_per_column.round(2))

# Convert columns to numeric if possible
df['goals'] = pd.to_numeric(df['goals'], errors='coerce')
df['assists'] = pd.to_numeric(df['assists'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['position'] = df['position'].astype(str)

# Fill missing values
df['position'].fillna('Unknown', inplace=True)
df['position_encoded'] = df['position'].astype('category').cat.codes

# Fill missing assists
goals_assist_avg = df.groupby('goals')['assists'].mean()
df['assists'] = df.apply(
    lambda row: goals_assist_avg.get(row['goals'], df['assists'].mean()) if pd.isnull(row['assists']) else row['assists'],
    axis=1
)

# Fill missing rating
grouped_avg = df.groupby(['goals', 'assists'])['rating'].mean()
df['rating'] = df.apply(
    lambda row: grouped_avg.get((row['goals'], row['assists']), df['rating'].mean()) if pd.isnull(row['rating']) else row['rating'],
    axis=1
)

# Fill missing club
if 'club' in df.columns:
    position_club_mode = df.groupby('position')['club'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    df['club'] = df.apply(
        lambda row: position_club_mode.get(row['position'], df['club'].mode()[0]) if pd.isnull(row['club']) else row['club'],
        axis=1
    )

# Drop rows with missing 'goals' or 'position'
df_filtered = df.dropna(subset=['goals', 'position'])

# 1. Bar Chart: Total Goals vs Position
plt.figure(figsize=(10, 6))
goals_by_position = df_filtered.groupby('position')['goals'].sum()
goals_by_position.plot(kind='bar', color='skyblue')
plt.title('Total Goals by Position', fontsize=14)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Total Goals', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Scatter Plot: Goals vs Age
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['goals'], alpha=0.6, color='green')
plt.title('Scatter Plot: Goals vs Age')
plt.xlabel('Age')
plt.ylabel('Goals')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Rating
plt.figure(figsize=(10, 6))
plt.hist(df['rating'].dropna(), bins=20, color='cornflowerblue', edgecolor='black')
plt.title('Histogram of Player Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Countplot: Number of Players by Age Group
df['age_group'] = pd.cut(df['age'], bins=[15, 20, 25, 30, 35, 40, 45], labels=['15-20', '21-25', '26-30', '31-35', '36-40', '41-45'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='age_group', palette='Set2')
plt.title('Number of Players by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 5. Pie Chart: Distribution of Players by Position
plt.figure(figsize=(8, 8))
position_counts = df['position'].value_counts()
plt.pie(position_counts, labels=position_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Player Distribution by Position')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 6. Heatmap: position vs rating
plt.figure(figsize=(8, 6))
correlation = df[['goals', 'assists', 'rating', 'age']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 7. Boxplot: Rating vs Age Group
df['age_group'] = pd.cut(df['age'], bins=[15, 20, 25, 30, 35, 40, 45], labels=['15-20', '21-25', '26-30', '31-35', '36-40', '41-45'])
plt.figure(figsize=(10, 6))
sns.boxplot(x='age_group', y='rating', data=df, palette='Pastel1')
plt.title('Boxplot of Rating by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

# Save cleaned dataset
df_cleaned = df.dropna()
df_cleaned.to_excel("cleaned_dataset.xlsx", index=False)
print("Cleaned dataset saved as 'cleaned_dataset.xlsx'")