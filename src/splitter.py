import pandas as pd
import logging

logger = logging.getLogger(__name__)


def Splitter(data: pd.DataFrame, year_col: str = "year"):
    """Split a time-series DataFrame into incremental train/val/test folds.

    Each fold adds one year to the training set, with a fixed validation
    window of one year and a fixed test window of the year immediately
    after. The minimum dataset length is 5 years: 3 for the first training
    fold, 1 for validation, and 1 for testing.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least a year column with consecutive,
        non-missing integer years.
    year_col : str, optional
        Name of the column containing year values, by default "year".

    Returns
    -------
    folds : dict of str to dict
        A dictionary where each key is a fold label (e.g. ``"Fold_1"``)
        and each value is a dict with three keys:

        - ``"train"`` : list of int — years used for training.
        - ``"val"``   : list of int — single year used for validation.
        - ``"test"``  : list of int — single year used for testing.

    Raises
    ------
    ValueError
        If fewer than 5 unique years are present in ``year_col``.
    ValueError
        If the years in ``year_col`` are not consecutive (i.e. gaps exist).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"year": range(2011, 2020), "feature": range(9)})
    >>> folds = Splitter(df)
    >>> folds["Fold_1"]
    {'train': [2011, 2012, 2013], 'val': [2014], 'test': [2015]}
    >>> folds["Fold_2"]
    {'train': [2011, 2012, 2013, 2014], 'val': [2015], 'test': [2016]}
    """

    folds = {}
    years = []

    years_dt = list(data[year_col].unique())
    years_dt = sorted(years_dt)
    for i in years_dt:
        years.append(int(i))
    len_years = len(years)
    expected_len_years = len(range(years[0], years[-1] + 1))

    if len_years <= 4:
        raise ValueError(
            f"There are not enough years in this dataset. This function needs a minimum of 5 years, found {len_years}."
        )

    if len_years != expected_len_years:
        missing_years = []
        for expected_year in range(years[0], years[-1] + 1):
            if expected_year not in years:
                missing_years.append(expected_year)
        raise ValueError(f"There years are missing: {missing_years}.")

    for i in range(len_years - 4):
        fold = f"Fold_{i + 1}"
        val_index = (
            i + 3
        )  # Also used for training fold, because the slice excludes the validation year
        test_index = i + 4
        folds[fold] = {
            "train": years[:val_index],
            "val": [years[val_index]],
            "test": [years[test_index]],
        }

    logger.info(f"{len(folds)} sets of folds were created.")

    return folds


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    d = {
        "key1": [10, 100.1, 0.98, 1.2, 3.9, 5.3, 8.7, 16.2, 32],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    }

    df = pd.DataFrame.from_dict(d, orient="index").transpose()
    # df["year"] = pd.to_datetime(df["year"], format="%Y").dt.year
    Splitter(data=df)
