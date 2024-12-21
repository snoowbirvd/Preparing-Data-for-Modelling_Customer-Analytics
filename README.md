# Customer Analytics: Preparing Data for Modeling

This repository contains the implementation and documentation of a data preprocessing project aimed at optimizing customer data for modeling tasks. The dataset, `customer_train.csv`, was processed according to the requirements provided by the Head Data Scientist at Training Data Ltd.

## Project Overview

The goal of this project is to create an efficient DataFrame called `ds_jobs_transformed` by performing the following tasks:

1. Optimize column data types to reduce memory usage.
2. Filter the dataset to retain only relevant rows.

The requirements for data transformation are as follows:

- **Two-factor categories** must be stored as Booleans.
- **Integer-only columns** must be stored as 32-bit integers.
- **Float columns** must be stored as 16-bit floats.
- **Nominal categorical data** must be stored as the `category` data type.
- **Ordinal categorical data** must be stored as ordered categories.
- Filter the DataFrame to only include rows meeting specific criteria.

By the end of the project, the dataset should be optimized for memory efficiency and tailored to the needs of experienced professionals at enterprise companies.

## Steps

### 1. Load the Dataset

The raw dataset is loaded using pandas, and a copy is created to avoid modifying the original data.

```python
import pandas as pd

ds_jobs = pd.read_csv("customer_train.csv")
ds_jobs_transformed = ds_jobs.copy()
```

---

### 2. Exploratory Data Analysis (EDA)

Performed EDA to identify the data types and unique values of categorical columns. This step helps differentiate between nominal and ordinal categories and informs the creation of transformation mappings.

```python
for col in ds_jobs.select_dtypes("object").columns:
    print(ds_jobs[col].value_counts(), '\n')
```

---

### 3. Define Mappings for Categorical Columns

Based on the EDA results:

- **Ordinal categories** were assigned a natural order using a dictionary.
- **Two-factor categories** were mapped to boolean values.

#### Ordinal Categories Mapping

```python
ordered_cats = {
    'enrolled_university': ['no_enrollment', 'Part time course', 'Full time course'],
    'education_level': ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
    'experience': ['<1'] + list(map(str, range(1, 21))) + ['>20'],
    'company_size': ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'],
    'last_new_job': ['never', '1', '2', '3', '4', '>4']
}
```

#### Two-Factor Categories Mapping

```python
two_factor_cats = {
    'relevant_experience': {'No relevant experience': False, 'Has relevant experience': True},
    'job_change': {0.0: False, 1.0: True}
}
```

---

### 4. Transform Columns Efficiently

A `for` loop was used to iterate over columns and apply the necessary transformations:

#### Steps:

1. **Two-Factor Categories**: Converted to boolean using mapping.
2. **Integer Columns**: Converted to `int32` for memory efficiency.
3. **Float Columns**: Converted to `float16` for memory efficiency.
4. **Ordered Categorical Columns**: Converted to ordered categories using the mapping dictionary.
5. **Nominal Categories**: Converted to `category` type.

#### Code:

```python
for col in ds_jobs_transformed:
    if col in two_factor_cats:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].map(two_factor_cats[col])
    elif col in ['student_id', 'training_hours']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('int32')
    elif col == 'city_development_index':
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('float16')
    elif col in ordered_cats.keys():
        category = pd.CategoricalDtype(ordered_cats[col], ordered=True)
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype(category)
    else:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('category')
```

---

### 5. Filter the Dataset

Filtered the dataset to retain only rows where:

- The `experience` column has 10 or more years.
- The `company_size` column indicates companies with at least 1000 employees.

#### Code:

```python
ds_jobs_transformed = ds_jobs_transformed[
    (ds_jobs_transformed['experience'] >= '10') &
    (ds_jobs_transformed['company_size'] >= '1000-4999')
]
```

---

### 6. Verify Results

Checked the memory usage and structure of the transformed DataFrame to ensure compliance with project requirements.

```python
print(ds_jobs.info())
print(ds_jobs_transformed.info())
```

---

## Results

- The transformed DataFrame (`ds_jobs_transformed`) has significantly reduced memory usage compared to the original dataset.
- The dataset contains only relevant rows based on the filtering criteria.

---

## Key Takeaways

1. **Categorical data optimization** dramatically reduces memory usage, especially for large datasets.
2. Proper EDA is crucial to correctly identify data types and create meaningful mappings for transformation.
3. Filtering rows based on specific criteria tailors the dataset to the intended audience or task.

---

## Future Improvements

1. **Automate the identification of ordinal categories**:
   - Leveraging natural language processing (NLP) techniques could allow for automated detection and ordering of categories, saving time and reducing manual effort.

2. **Implement advanced filtering logic**:
   - Allow dynamic user-defined thresholds for filtering instead of hardcoding criteria, making the process more adaptable to various use cases.

3. **Dynamic datatype inference**:
   - Incorporate machine learning or heuristic algorithms to dynamically infer the most memory-efficient data types for large and complex datasets.

---

## How to Use This Repository

1. Clone the repository.
2. Place the `customer_train.csv` file in the same directory as the script.
3. Run the Python script to generate the `ds_jobs_transformed` DataFrame.


