import pandas as pd
import logging
import numpy as np


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


class WeatherPrep:
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
        data : dataset is loaded into the class attributes.
        """

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.info("A long time ago in a galaxy far, far away....")
        logging.info("The process of data preparation was started.")

        self.data = pd.read_csv(
            filepath_or_buffer=data_path, sep=separator, skiprows=skip_rows
        )
        logging.info(
            f"New instance of a class is constructed, shape is {self.data.shape}."
        )

    def clean_data(self):
        self.data = self.data.rename(columns=lambda x: x.strip())

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

        logging.debug("Checking features...")
        for k in new_col_names.keys():
            if k not in self.data.columns:
                raise NameError(f"Column {k} not found in the data.")
        logging.debug("All necessary features are in the dataset.")

        logging.debug("Checking datatypes...")
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="raise")

        logging.debug("All datatypes are valid.")

        self.data = self.data.rename(columns=new_col_names)

        if self.data.isna().sum().sum() != 0:
            raise ValueError(
                f"Found {self.data.isna().sum().sum()} missing values, expected 0. Please inspect your dataset."
            )

        self.data["date"] = pd.to_datetime(self.data["date"], format="%Y%m%d")

        logging.info(f"Data cleaning completed. Shape is {self.data.shape}")

    def feature_engineering(self, window_length: list[int]):

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

        logging.debug("Updated scales.")

        # Create a new column to weigh how close the mean vector wind speed is to the
        # overall daily wind speed.
        self.data["windVectorWeight"] = round(
            self.data["windVectorAvgSpeed"] / self.data["windDailyAvgSpeed"].replace(0, np.nan), 3
        )

        # Use custom function to convert information from a wind vector in degrees
        # (0 to 360) to proportions in every wind direction: North, East, South,
        # and West.
        self.data[["north", "east", "south", "west"]] = pd.DataFrame(
            self.data["windVectorDirection"]
            .apply(wind_direction_converter)
            .tolist()
        )

        # Adjusts the values using windVectorWeight
        for col in self.data[["north", "east", "south", "west"]]:
            self.data[col] = round(self.data[col] * self.data["windVectorWeight"], 3)

        self.data = self.data.drop(columns="windVectorDirection")

        logging.debug("Converted wind vector to proportions.")

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

        logging.debug("Created rolling window features.")

        # Remove the year 2010 and reset index.
        self.data = self.data[self.data["date"] >= "2011-01-01"].reset_index(drop=True)

        logging.debug(f"The dataset has the following columns {self.data.columns}.")
        logging.info(f"Feature engineering completed. Shape is {self.data.shape}.")


if __name__ == "__main__":
    p = WeatherPrep(
        data_path="data/raw/weather_data_original.txt", separator=",", skip_rows=22
    )
    p.clean_data()
    p.feature_engineering(window_length=[3, 7])
    
