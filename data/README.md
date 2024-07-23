# Data
Dataset Description

The high-level goal of this project is to detect, analyze, and understand UHIs by leveraging data mining and machine learning to understand the microclimatic effects of urbanization in various metropolitan landscapes.

Climatological and hourly time-series data has been collected from several weather stations run by the National Weather Service (NWS), Federal Aviation Administration (FAA), and Department of Defense (DOD), which encompass data collected from airports and other significant areas. This data has been divided into three Local Climatological Datasets (LCD) for Arlington, Dallas, and Denton for the year 2022.


# Codebook for Dataset
The dataset gathered from National Center for Environmental Information (NCEI) (https://www.ncei.noaa.gov/) offers these key hourly summaries that I will be analyzing.

## Variable Names and Descriptions:
| Column Name                        | Description                                                                                                               | Data Type    |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------|
| DATE                               | Date of the observation (year, month, and day) along with time of observation given as a 4-digit number using a 24-hour clock in local standard time | String       |
| HourlyAverageDryBulbTemperature    | Standard air temperature reported, given in whole degrees Fahrenheit                                                        | Float        |
| HourlyAverageWetBulbTemperature    | Wet-bulb temperature, given in whole degrees Fahrenheit                                                                    | Float        |
| HourlyAverageDewPointTemperature   | Dew point temperature, given in whole degrees Fahrenheit                                                                   | Float        |
| HourlyPrecipitation                 | Total Liquid Content (TLC) water equivalent amount of precipitation for the day                                            | Float        |
| HourlyAverageSeaLevelPressure       | Sea level pressure given in inches of Mercury (in Hg)                                                                     | Float        |
| HourlyAverageStationPressure        | Atmospheric pressure observed at the station during the time of observation, given in inches of Mercury (in Hg)          | Float        |
| HourlyAverageRelativeHumidity       | Relative humidity given to the nearest whole percentage                                                                   | Float        |
| HourlyAverageWindSpeed              | Speed of the wind at the time of observation given in miles per hour (mph)                                                | Float        |
| HourlySustainedWindDirection       | Wind direction from true north using compass directions (e.g., 360 = true north, 180 = south, 270 = west, etc.)           | Float        |
