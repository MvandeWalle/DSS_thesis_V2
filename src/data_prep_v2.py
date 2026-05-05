import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)


def wind_direction_converter(degrees: int):
    """Converts vector to proportions in every wind direction.

    Parameters
    ----------
    degrees : a wind vector in degrees, corresponding to the degrees on a
    compass (360=north; 90=east; 180=south; 270=west; 0=calm/variable).

    Returns
    -------
    north, east, south, west : proportions in every wind direction.

    Example
    -----
    The value 45 (north-east) would return the following:
    north : 0.5
    east : 0.5
    south : 0
    west: 0
    """

    north = 0
    east = 0
    south = 0
    west = 0
    if 0 < degrees <= 90:
        north = (90 - degrees) / 90
        east = degrees / 90
    elif 90 < degrees <= 180:
        degrees = degrees - 90
        east = (90 - degrees) / 90
        south = degrees / 90
    elif 180 < degrees <= 270:
        degrees = degrees - 180
        south = (90 - degrees) / 90
        west = degrees / 90
    elif 270 < degrees <= 360:
        degrees = degrees - 270
        west = (90 - degrees) / 90
        north = degrees / 90
    north = round(north, 3)
    east = round(east, 3)
    south = round(south, 3)
    west = round(west, 3)
    return north, east, south, west


def rolling_windows(
    data: pd.DataFrame, method: str, columns: list[str], length: list[int], decimal: int
):
    """Creates summary statistics over sliding windows.

    Parameters
    ----------
    data : a pandas DataFrame.

    method : a string mentioning 'mean' or 'sum'.

    columns : a list of column names from the corresponding DataFrame.

    length : a list containing different values for the length of the
    sliding windows.

    decimal : an integer to mention how many decimals the new columns should
    have

    Returns
    -------
    data : modified version of the input DataFrame. New columns have
    been added, which contain summary statistics over sliding windows."""

    allowed_methods = ["mean", "sum"]
    if method not in allowed_methods:
        raise ValueError(f"Method must be '{allowed_methods}', got '{method}' instead.")

    elif method == "mean":
        for col in columns:
            for value in length:
                new_name = col + "_mean_" + str(value)
                data[new_name] = data[col].rolling(value).mean().round(decimal)
                if decimal == 0:
                    data[new_name] = data[new_name].fillna(0).astype("int")

    elif method == "sum":
        for col in columns:
            for value in length:
                new_name = col + "_sum_" + str(value)
                data[new_name] = data[col].rolling(value).sum().round(decimal)
                if decimal == 0:
                    data[new_name] = data[new_name].fillna(0).astype("int")

    return data


class WeatherPrep():
    """This class prepares the weather data"""

    def __init__(self, data_path: str, separator: str = ",", skip_rows: int = 0):
        """Initialise class, configure logger and load the data.

        Parameters
        ----------
        data_path : path to the dataset (.csv or .txt).

        separator : choose the separator in the dataset (comma or semicolon). Default is ','.

        skip_rows : if the file contains metadata before the dataset itself, mention how many rows so these can be skipped when reading the file. Default is 0.

        New attributes
        --------------
        data : weather dataset is loaded into the class attributes.
        """

        logger.debug("The process of data preparation was started.")

        self.data = pd.read_csv(
            filepath_or_buffer=data_path, sep=separator, skiprows=skip_rows
        )
        logger.debug(f"WeatherPrep is constructed, data shape is {self.data.shape}.")

    def clean_data(self):
        # Removing trailing space in column names.
        self.data = self.data.rename(columns=lambda x: x.strip())

        # If there's only one weather station, this column can be dropped.
        if self.data["# STN"].nunique() == 1:
            self.data = self.data.drop("# STN", axis=1)

        # Changing column names for clarity.
        new_col_names = {
            "YYYYMMDD": "date",
            "DDVEC": "windVectorDirection",
            "FHVEC": "windVectorAvgSpeed",
            "FG": "windDailyAvgSpeed",
            "FHX": "windHourlyMaxSpeed",
            "FHN": "windHourlyMinSpeed",
            "TG": "tempDailyAvg",
            "TN": "tempDailyMin",
            "TX": "tempDailyMax",
            "RH": "precipDailySum",
            "PG": "airpresDailyAvg",
            "PX": "airpresHourlyMax",
            "PN": "airpresHourlyMin",
            "UG": "humidDailyAvg",
            "UX": "humidDailyMax",
            "UN": "humidDailyMin",
        }

        # Checking if all necessary features are in the dataset
        logger.debug("Checking features...")
        for k in new_col_names.keys():
            if k not in self.data.columns:
                raise NameError(f"Column {k} not found in the data.")
        logger.debug("All necessary features are in the dataset.")

        # Checking if all column are numeric, if not, try to convert them.
        logger.debug("Checking datatypes...")
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="raise")

        logger.debug("All datatypes are valid.")

        # Rename the columns
        self.data = self.data.rename(columns=new_col_names)

        # Change date format in every dataset, so they have the same format for merging
        self.data["date"] = pd.to_datetime(self.data["date"], format="%Y%m%d")

        # Check for missing values.
        if self.data.isna().sum().sum() != 0:
            raise ValueError(
                f"Found {self.data.isna().sum().sum()} missing values, expected 0. Please inspect your dataset."
            )

        logger.debug(f"WeatherPrep: data is cleaned. Shape is {self.data.shape}")

    def feature_engineering(self, window_length: list[int]):
        """Apply feature engineering to the cleaned weather dataset.

        Scales raw KNMI variables to their correct units, computes a wind
        vector weight, decomposes wind direction into cardinal proportions,
        and adds rolling window summary statistics for temperature and
        precipitation.

        Parameters
        ----------
        window_length : list of int
            Window sizes in days over which to compute rolling statistics.
            For example, ``[3, 7, 28]`` creates 3-day, 7-day, and 28-day
            rolling means and sums. Must be called after ``clean_data()``.

        Returns
        -------
        None
            Modifies ``self.data`` in place.

        Examples
        --------
        >>> wp = WeatherPrep("weather.csv", separator=",", skip_rows=22)
        >>> wp.clean_data()
        >>> wp.feature_engineering(window_length=[3, 7, 28])
        """

        # Update scales
        wrong_scale = [
            "windVectorAvgSpeed",
            "windDailyAvgSpeed",
            "windHourlyMaxSpeed",
            "windHourlyMinSpeed",
            "tempDailyAvg",
            "tempDailyMin",
            "tempDailyMax",
            "precipDailySum",
        ]

        self.data[wrong_scale] = self.data[wrong_scale] / 10

        logger.debug("Updated scales.")

        # Create a new column to weigh how close the mean vector wind speed is to the
        # overall daily wind speed.
        self.data["windVectorWeight"] = round(
            self.data["windVectorAvgSpeed"]
            / self.data["windDailyAvgSpeed"].replace(0, np.nan),
            3,
        )

        # Use custom function to convert information from a wind vector in degrees
        # (0 to 360) to proportions in every wind direction: North, East, South,
        # and West.
        self.data[["north", "east", "south", "west"]] = pd.DataFrame(
            self.data["windVectorDirection"].apply(wind_direction_converter).tolist()
        )

        # Adjusts the values using windVectorWeight
        for col in self.data[["north", "east", "south", "west"]]:
            self.data[col] = round(self.data[col] * self.data["windVectorWeight"], 3)

        # The original column can now be dropped.
        self.data = self.data.drop(columns="windVectorDirection")

        logger.debug("Converted wind vector to proportions.")

        # Rolling windows for temperature (mean)
        self.data = rolling_windows(
            data=self.data,
            method="mean",
            length=window_length,
            decimal=1,
            columns=["tempDailyAvg", "tempDailyMin", "tempDailyMax"],
        )

        # Rolling windows for airpressure and humidity (mean)
        self.data = rolling_windows(
            data=self.data,
            method="mean",
            length=window_length,
            decimal=0,
            columns=["airpresDailyAvg", "humidDailyAvg"],
        )

        # Rolling windows for precipitation (sum)
        self.data = rolling_windows(
            data=self.data,
            method="sum",
            length=window_length,
            decimal=0,
            columns=["precipDailySum"],
        )

        logger.debug("Created rolling window features.")

        # Remove the year 2010 and reset index.
        self.data = self.data[self.data["date"] >= "2011-01-01"].reset_index(drop=True)

        logger.debug(
            f"WeatherPrep: The dataset has the following columns {self.data.columns}."
        )
        logger.debug(
            f"WeatherPrep: feature engineering completed. Shape is {self.data.shape}."
        )

        return self.data

    def write_file(self, folder: str):
        datapath = f"{folder}/weather_data_processed.csv"
        self.data.to_csv(datapath, sep=",", index=False)
        logger.debug(
            f"WeatherPrep: file saved at {datapath} with shape {self.data.shape}."
        )
        return datapath


class WildfirePrep():
    """This class prepares the wildfire data"""

    def __init__(self, data_path: str, separator: str = ",", skip_rows: int = 0):
        """Initialise class, configure logger and load the data.

        Parameters
        ----------
        data_path : path to the dataset (.csv or .txt).

        separator : choose the separator in the dataset (comma or semicolon). Default is ','.

        skip_rows : if the file contains metadata before the dataset itself, mention how many rows so these can be skipped when reading the file. Default is 0.

        New attributes
        --------------
        data : wildfire dataset is loaded into the class attributes.
        """

        logger.debug("The process of data preparation was started.")

        self.data = pd.read_csv(
            filepath_or_buffer=data_path, sep=separator, skiprows=skip_rows
        )
        logger.debug(f"WildfirePrep is constructed, data shape is {self.data.shape}.")

    def clean_data(self):
        logger.debug("Wildfire data cleaning initiated.")

        # Change date format in every dataset, so they have the same format for merging
        self.data = self.data.rename(columns={"#DateId": "date"})
        self.data["date"] = pd.to_datetime(self.data["date"], format="%Y%m%d")

        # Sort values and filter for 2011-2025
        self.data = self.data.sort_values("date").reset_index(drop=True)
        self.data = self.data[
            (self.data["date"].dt.year >= 2011) & (self.data["date"].dt.year <= 2025)
        ]

        logger.debug(f"WildfirePrep: data is cleaned. Shape is {self.data.shape}")

    def feature_engineering(self):
        # Obtain the wildfire counts per date
        self.wildfire_dates = (
            self.data.groupby("date").size().reset_index(name="numeric_wf")
        )

        # For the binary wildfire variable, all the values should be 1, because in this dataset there are only dates on which wildfires occurred.
        self.wildfire_dates["binary_wf"] = 1

        return self.wildfire_dates

    def write_file(self, folder: str):
        output_datapath = f"{folder}/wildfire_processed.csv"

        self.wildfire_dates.to_csv(output_datapath, sep=",", index=False)
        logger.debug(
            f"WildfirePrep: file saved at {output_datapath} with shape {self.wildfire_dates.shape}."
        )

        return output_datapath


class CalendarPrep():
    """This class prepares the calendar data"""

    def __init__(self, data_path: str, separator: str = ",", skip_rows: int = 0):
        """Initialise class and load the data.

        Parameters
        ----------
        data_path : path to the dataset (.csv or .txt).

        separator : choose the separator in the dataset (comma or semicolon). Default is ','.

        skip_rows : if the file contains metadata before the dataset itself, mention how many rows so these can be skipped when reading the file. Default is 0.

        New attributes
        --------------
        data : calendar dataset is loaded into the class attributes.
        """

        logger.debug("The process of data preparation was started.")

        self.data = pd.read_csv(
            filepath_or_buffer=data_path, sep=separator, skiprows=skip_rows
        )

        logger.debug(f"CalendarPrep is constructed, data shape is {self.data.shape}.")

    def clean_data(self):
        logger.debug("CalendarPrep: data cleaning initiated")

        # Change date format in every dataset, so they have the same format for merging
        self.data["date"] = pd.to_datetime(self.data["date"], format="%d-%m-%Y")

        # Dropping irrelevant and categorical columns
        self.data = self.data.drop(
            columns=[
                "#DateId",
                "dagvhJaar",
                "NaamVakNoord",
                "NaamVakMidden",
                "NaamVakZuid",
                "NaamVakBE",
                "NaamVakDE",
                "feestdagenVakantie_BE",
                "feestdagenVakantie_DE",
                "dagvdWeekOmschr",
                "feestdagNaam",
                "_catDagsoort",
            ]
        )

        # Rename the columns
        new_col_names = {
            "_isFeestdagNL": "holiday_NL",
            "_isWerkdag": "workday",
            "VakantieNoord": "vacation_NL_North",
            "VakantieMidden": "vacation_NL_Central",
            "VakantieZuid": "vacation_NL_South",
            "VakantieVlaanderen": "vacation_BE",
            "VakantieNordrheinWF": "vacation_GE",
        }

        self.data = self.data.rename(columns=new_col_names)

        # I need to convert every column (except date) to Boolean, so it's saved properly when converting to CSV
        binary_cols = [
            "holiday_NL",
            "workday",
            "vacation_NL_North",
            "vacation_NL_Central",
            "vacation_NL_South",
            "vacation_BE",
            "vacation_GE",
        ]

        # Filling the NaN with 0 and then converting to integer
        self.data[binary_cols] = self.data[binary_cols].fillna(0).astype("int")
        logger.debug(f"CalendarPrep: data is cleaned. Shape is {self.data.shape}.")

    def write_file(self, folder: str):
        datapath = f"{folder}/calendar_data_processed.csv"
        self.data.to_csv(datapath, sep=",", index=False)
        logger.debug(
            f"CalendarPrep: file saved at {datapath} with shape {self.data.shape}."
        )
        return datapath


def DataMerger(
    weather_path: str,
    calendar_path: str,
    wildfire_path: str,
    output_folder: str,
):
    """Merge processed weather, calendar, and wildfire datasets and save splits.

    Loads the three processed datasets, validates their structure, and
    merges them on the ``date`` column. The wildfire target columns are
    encoded as binary (int; 0/1) and numeric (int). The merged dataset is then split chronologically into a training set (2011–2023) and an out-of-sample test set (2024–2025),
    both of which are written to CSV. Example data is also created (2011-2015).

    Note: the wildfire dataset may contain multiple fire records per day
    by design, so duplicate date checks are not applied to it.

    Parameters
    ----------
    weather_path : str
        Path to the processed weather CSV file.
    calendar_path : str
        Path to the processed calendar CSV file.
    wildfire_path : str
        Path to the processed wildfire CSV file, expected to contain
        a ``count`` column with the number of fires per day.
    output_folder : str
        Directory where the output CSV files will be saved.

    Returns
    -------
    Three datasets: example, train, and test. These are also written to csv file.


    Raises
    ------
    KeyError
        If any of the three datasets is missing a ``date`` column.
    ValueError
        If the weather or calendar datasets contain duplicate dates.
    ValueError
        If the weather–calendar merge results in an empty dataset.

    Examples
    --------
    >>> DataMerger(
    ...     weather_path="data/processed/weather_data_processed.csv",
    ...     calendar_path="data/processed/calendar_data_processed.csv",
    ...     wildfire_path="data/processed/wildfire_processed.csv",
    ...     output_folder="data/processed",
    ... )
    """

    # Load the files
    weather = pd.read_csv(weather_path)
    calendar = pd.read_csv(calendar_path)
    wildfire = pd.read_csv(wildfire_path)

    for name, df in {
        "weather": weather,
        "calendar": calendar,
        "wildfire": wildfire,
    }.items():
        if "date" not in df.columns:
            raise KeyError(f"{name} dataset missing 'date' column.")

        # Not really necessary for wildfire, but easier to keep one loop for everything.
        if df["date"].duplicated().any():
            raise ValueError(f"{name} dataset contains duplicate dates.")

    # Merge the weather and calendar datasets
    if weather.shape[0] != calendar.shape[0]:
        logger.warning(
            f"Weather and calendar datasets do not have the same length: {weather.shape[0]} vs {calendar.shape[0]}. A left join will be performed instead of a inner join. Please check the dataset."
        )
        dataset = weather.merge(calendar, how="left", on="date")

    else:
        logger.debug(
            f"Shape: weather ({weather.shape}) vs calendar ({calendar.shape})."
        )
        dataset = weather.merge(calendar, how="inner", on="date")

    if dataset.empty:
        raise ValueError("Merge resulted in empty dataset. Check date alignment.")

    dataset = dataset.merge(wildfire, how="left", on="date")
    logger.debug(
        f"Merger: the three datasets are merged and the current dataset has shape {dataset.shape}."
    )

    # Add a year column for easier splitting later
    dataset["date"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d")
    dataset["year"] = dataset["date"].dt.year

    # Fill the NaN values, because on those dates no wildfires occurred.
    dataset["numeric_wf"] = dataset["numeric_wf"].fillna(0).astype(int)
    dataset["binary_wf"] = dataset["binary_wf"].fillna(0).astype(int)

    # Example data
    example_data = dataset[
        (dataset["date"] >= "2011-01-01") & (dataset["date"] <= "2015-12-31")
    ]

    example_path = f"{output_folder}/example_dataset.csv"
    example_data.to_csv(example_path, sep=",", index=False)
    logger.debug(
        f"Merger: example dataset saved at {example_path} with shape {example_data.shape}."
    )

    # For training and validating
    train_data = dataset[
        (dataset["date"] >= "2011-01-01") & (dataset["date"] <= "2023-12-31")
    ]

    train_path = f"{output_folder}/train_dataset.csv"
    train_data.to_csv(train_path, sep=",", index=False)
    logger.debug(
        f"Merger: train dataset saved at {train_path} with shape {train_data.shape}."
    )

    # For out-of-sample testing
    test_data = dataset[
        (dataset["date"] >= "2024-01-01") & (dataset["date"] <= "2025-12-31")
    ]

    test_path = f"{output_folder}/test_dataset.csv"
    test_data.to_csv(test_path, sep=",", index=False)
    logger.debug(
        f"Merger: test dataset saved at {test_path} with shape {test_data.shape}."
    )

    logger.info(f"Data Merger completed. Datasets are saved in {output_folder}")

    return example_data, train_data, test_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    a = WeatherPrep(
        data_path="data/raw/weather_data_original.txt", separator=",", skip_rows=22
    )
    a.clean_data()
    a.feature_engineering(window_length=[3, 7])
    a.write_file(folder="data/processed")

    b = WildfirePrep(data_path="data/raw/wildfire_data.csv", separator=";", skip_rows=0)
    b.clean_data()
    b.feature_engineering()
    b.write_file(folder="data/processed")

    c = CalendarPrep(
        data_path="data/raw/calendar_features.csv", separator=";", skip_rows=0
    )
    c.clean_data()
    c.write_file(folder="data/processed")

    example, train, test = DataMerger(
        weather_path="data/processed/weather_data_processed.csv",
        calendar_path="data/processed/calendar_data_processed.csv",
        wildfire_path="data/processed/wildfire_processed.csv",
        output_folder="data/processed",
    )

    print(train.head(30))
