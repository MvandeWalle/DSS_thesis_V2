import pandas as pd

d = {
    "key1": [10, 100.1, 0.98, 1.2, 3.9, 5.3, 8.7, 16.2, 32],
    "key2": [72.5],
    "year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df["year"] = pd.to_datetime(df["year"], format="%Y").dt.year


def Splitter(data: pd.DataFrame, year_col: str = "year"):
    folds = {}
    years = []

    years_dt = list(data[year_col].unique())
    years_dt = sorted(years_dt)
    for i in years_dt:
        years.append(int(i))
    len_years = len(years)
    expected_len_years = len(range(years[0], years[-1] + 1))

    if len_years <= 4:
        raise ValueError(f"There are not enough years in this dataset. This function needs a minimum of 5 years, found {len_years}.")

    if len_years != expected_len_years:
        missing_years = []
        for expected_year in range(years[0], years[-1] + 1):
            if expected_year not in years:
                missing_years.append(expected_year)
        raise ValueError(f"There years are missing: {missing_years}.")

    for i in range(len_years - 4):
        fold = f"Fold_{i + 1}"
        val_index = i + 3   # Also used for training fold, because the slice excludes the validation year
        test_index = i + 4
        folds[fold] = {'train': years[:val_index], 'val': [years[val_index]], 'test': [years[test_index]]}
    return folds


mara = 1

if mara == 1:
    print(Splitter(data=df))

if mara == 2:
    x, y, z = Splitter(data=df)
    print(x, y, z)
    for i in [1, 2, 3, 4, 5]:
        fold = f"Fold_{i}"
        print(f"This is {fold}!")
        print("Training set")
        print(df[df['year'].isin(x[fold])])
        print("Validation set")
        print(df[df['year'].isin(y[fold])])
        print("Test")
        print(df[df['year'].isin(z[fold])])

