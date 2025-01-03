# Customer Analytics: Preparing Data for Modeling
Dataset, `customer_train.csv`
Implementation and documentation of a data preprocessing project aimed at optimizing customer data for modeling tasks.

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
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19158 entries, 0 to 19157
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   student_id              19158 non-null  int64  
 1   city                    19158 non-null  object 
 2   city_development_index  19158 non-null  float64
 3   gender                  14650 non-null  object 
 4   relevant_experience     19158 non-null  object 
 5   enrolled_university     18772 non-null  object 
 6   education_level         18698 non-null  object 
 7   major_discipline        16345 non-null  object 
 8   experience              19093 non-null  object 
 9   company_size            13220 non-null  object 
 10  company_type            13018 non-null  object 
 11  last_new_job            18735 non-null  object 
 12  training_hours          19158 non-null  int64  
 13  job_change              19158 non-null  float64
dtypes: float64(2), int64(2), object(10)
memory usage: 2.0+ MB
```
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
```
city unique values:
 city_103    4355
city_21     2702
city_16     1533
city_114    1336
city_160     845
            ... 
city_129       3
city_111       3
city_121       3
city_140       1
city_171       1
Name: city, Length: 123, dtype: int64 

gender unique values:
 Male      13221
Female     1238
Other       191
Name: gender, dtype: int64 

relevant_experience unique values:
 Has relevant experience    13792
No relevant experience      5366
Name: relevant_experience, dtype: int64 

enrolled_university unique values:
 no_enrollment       13817
Full time course     3757
Part time course     1198
Name: enrolled_university, dtype: int64 

education_level unique values:
 Graduate          11598
Masters            4361
High School        2017
Phd                 414
Primary School      308
Name: education_level, dtype: int64 

major_discipline unique values:
 STEM               14492
Humanities           669
Other                381
Business Degree      327
Arts                 253
No Major             223
Name: major_discipline, dtype: int64 

experience unique values:
 >20    3286
5      1430
4      1403
3      1354
6      1216
2      1127
7      1028
10      985
9       980
8       802
15      686
11      664
14      586
1       549
<1      522
16      508
12      494
13      399
17      342
19      304
18      280
20      148
Name: experience, dtype: int64 

company_size unique values:
 50-99        3083
100-499      2571
10000+       2019
10-49        1471
1000-4999    1328
<10          1308
500-999       877
5000-9999     563
Name: company_size, dtype: int64 

company_type unique values:
 Pvt Ltd                9817
Funded Startup         1001
Public Sector           955
Early Stage Startup     603
NGO                     521
Other                   121
Name: company_type, dtype: int64 

last_new_job unique values:
 1        8040
>4       3290
2        2900
never    2452
4        1029
3        1024
Name: last_new_job, dtype: int64
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
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19158 entries, 0 to 19157
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   student_id              19158 non-null  int64  
 1   city                    19158 non-null  object 
 2   city_development_index  19158 non-null  float64
 3   gender                  14650 non-null  object 
 4   relevant_experience     19158 non-null  object 
 5   enrolled_university     18772 non-null  object 
 6   education_level         18698 non-null  object 
 7   major_discipline        16345 non-null  object 
 8   experience              19093 non-null  object 
 9   company_size            13220 non-null  object 
 10  company_type            13018 non-null  object 
 11  last_new_job            18735 non-null  object 
 12  training_hours          19158 non-null  int64  
 13  job_change              19158 non-null  float64
dtypes: float64(2), int64(2), object(10)
memory usage: 2.0+ MB
None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2201 entries, 9 to 19143
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype   
---  ------                  --------------  -----   
 0   student_id              2201 non-null   int32   
 1   city                    2201 non-null   category
 2   city_development_index  2201 non-null   float16 
 3   gender                  1821 non-null   category
 4   relevant_experience     2201 non-null   bool    
 5   enrolled_university     2185 non-null   category
 6   education_level         2184 non-null   category
 7   major_discipline        2097 non-null   category
 8   experience              2201 non-null   category
 9   company_size            2201 non-null   category
 10  company_type            2144 non-null   category
 11  last_new_job            2184 non-null   category
 12  training_hours          2201 non-null   int32   
 13  job_change              2201 non-null   bool    
dtypes: bool(2), category(9), float16(1), int32(2)
memory usage: 69.5 KB
None
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


