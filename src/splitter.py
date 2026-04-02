import pandas as pd

d = {
    "key1": [10, 100.1, 0.98, 1.2],
    "key2": [72.5],
    "year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
}

df = pd.DataFrame.from_dict(d, orient="index").transpose()
df["year"] = pd.to_datetime(df["year"], format="%Y").dt.year


def Splitter(data: pd.DataFrame, year_col: str = "year"):
    train_mask = {}
    val_mask = {}
    test_mask = {}
    count = 3
    years = []

    data[year_col] = pd.to_datetime(data[year_col], format="%Y").dt.year
    years_dt = list(data[year_col].unique())
    for i in years_dt:
        years.append(int(i))
    len_years = len(years)

    for i in range(len_years - 4):
        fold = f"Fold_{i + 1}"
        train_mask[fold] = years[:count]
        val_mask[fold] = years[count : (count + 1)]
        test_mask[fold] = years[(count + 1) : (count + 2)]
        count += 1
    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    print(Splitter(data=df))
