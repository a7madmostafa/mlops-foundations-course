# Course Data

This folder holds the flight-delay dataset you use across every lesson.
It is not committed to the repo — you download it from Kaggle so the
repo stays lean and anyone can reproduce the course from a clean clone.

## What's here

`flight_delays_YYYY_MM.csv` — one file per calendar month of 2025,
12 files total. Every lesson reads from **one shared pool** at the repo
root; data is never copied into lesson folders.

## How to get it

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle         # one-time
kaggle datasets download -d a7madmostafa/us-flight-delays-2025-bts-on-time-performance
unzip us-flight-delays-2025-bts-on-time-performance.zip -d data
```

### Option B — Kaggle website

Go to
<https://www.kaggle.com/datasets/a7madmostafa/us-flight-delays-2025-bts-on-time-performance>,
click **Download**, and unzip the files into the `data/` folder at the
repo root.

## Where the data comes from

The data is the US DOT Bureau of Transportation Statistics (BTS)
**Reporting Carrier On-Time Performance** database — public,
US-government open data.

### Re-download from BTS yourself (optional)

You can grab the raw data directly from the official source:
US DOT BTS → TranStats → On-Time.

- Field picker: <https://www.transtats.bts.gov/DL_SelectFields.aspx>
- Direct per-month zip:
  `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_1.zip`

The field picker always returns the complete file (110+ columns)
regardless of which boxes you tick — BTS does not trim columns. The
40-column schema used in this course is produced by trimming the full
download after download.

## Schema (40 columns, this exact order)

```
Year, Quarter, Month, DayofMonth, DayOfWeek, FlightDate,
Reporting_Airline, Flight_Number_Reporting_Airline, Origin,
OriginCityName, OriginState, Dest, DestCityName, DestState,
CRSDepTime, DepTime, DepDelay, DepDelayMinutes, DepDel15, TaxiOut,
WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrTime, ArrDelay,
ArrDelayMinutes, ArrDel15, Cancelled, CancellationCode, Diverted,
CRSElapsedTime, ActualElapsedTime, AirTime, Distance, CarrierDelay,
WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay
```

This is the schema used everywhere in the course. Lesson 1.1 starts
you on `flight_delays_2025_01.csv` (January 2025, ~540K rows); later
lessons pull in additional months.
