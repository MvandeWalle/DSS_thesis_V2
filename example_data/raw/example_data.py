# Load the dataset, take a subset and save as example data
## Using read and write methods to keep the original formatting
## Weather
with open("./data/raw/weather_data_original.txt", "r") as file:
    weather = file.readlines()
    weather = weather[:50]
    with open("example_data/raw/weather_example.txt", "w") as writer:
        writer.writelines(weather)

## Wildfire
with open("./data/raw/wildfire_data.csv", "r") as file:
    wf = file.readlines()
    wf = wf[:20]
    with open("example_data/raw/wildfire_example.csv", "w") as writer:
        writer.writelines(wf)

## Calendar
with open("./data/raw/calendar_features.csv", "r") as file:
    cal = file.readlines()
    cal = cal[:50]
    with open("example_data/raw/calendar_example.csv", "w") as writer:
        writer.writelines(cal)
