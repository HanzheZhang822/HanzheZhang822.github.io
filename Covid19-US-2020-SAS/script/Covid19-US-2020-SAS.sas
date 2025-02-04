/* Step 1: Import and Read Data */
PROC IMPORT DATAFILE='/home/u64112368/covidproject/us_counties_covid19_daily.csv' 
    OUT=us_covid_raw
    DBMS=CSV REPLACE;
    GETNAMES=YES;
    DATAROW=2;
RUN;

/* Step 2: Analyze Missing Values */
PROC MEANS DATA=us_covid_raw N NMISS MEAN MIN MAX;
    VAR cases deaths fips;
    TITLE "Missing Value Analysis - COVID-19 Raw Data";
RUN;

/* Identify counties with missing FIPS values */
data missingfips;
	set us_covid_raw;
	where fips=.;
run;
/* Some counties with missing FIPS values are labeled as 'Unknown'. */

/* Visualization of the cumulative cases trend for Unknown County */
PROC SGPANEL DATA=us_covid_raw;
    WHERE county = "Unknown";
    PANELBY state / ROWS=3 COLUMNS=3; 
    SERIES X=date Y=cases / LINEATTRS=(PATTERN=SOLID);
    TITLE "Cumulative Cases Trending for Unknown County of Each State";
    COLAXIS LABEL="Date" GRID;
    ROWAXIS LABEL="Cumulative Cases" GRID;
RUN;
/* The chart shows that in many states, the cases/deaths for Unknown County are not cumulative. */

/* Step 3: Exclude Unknown County Data */
DATA us_covid_raw;
    SET us_covid_raw;
    WHERE county NE "Unknown";
RUN;

/* Sort the data by state, county, and date */
PROC SORT DATA=us_covid_raw;
    BY state county date;
RUN;

/* Step 4: Handle missing values - Fill in missing data for each county before summarizing at the state level */
DATA us_covid_raw;
    SET us_covid_raw;
    BY state county date;

    /* Retain previous values */
    RETAIN last_cases last_deaths;

    /* Initialize for the first record of each county */
    IF FIRST.county THEN DO;
        last_cases = .;
        last_deaths = .;
    END;

    /* Handle missing values or inconsistent data (cumulative data should not decrease) */
    IF cases = . OR cases < last_cases THEN cases = last_cases;
    ELSE last_cases = cases;

    IF deaths = . OR deaths < last_deaths THEN deaths = last_deaths;
    ELSE last_deaths = deaths;
RUN;

/* Step 5: Data Cleaning - Aggregate data at the state level using PROC SUMMARY */
PROC SUMMARY DATA=us_covid_raw NWAY;
    CLASS state date; 
    VAR cases deaths; 
    OUTPUT OUT=us_covid_data_cleaned (DROP=_TYPE_ _FREQ_)
        SUM=cases deaths; 
RUN;

/* Step 6: Calculate Daily New Cases and Deaths */
DATA us_covid_daily;
    SET us_covid_data_cleaned;
    BY state date;
    
    /* Compute daily new cases and deaths */
    new_cases = cases - LAG(cases);
    new_deaths = deaths - LAG(deaths);
    
    /* Avoid errors from cross-state calculations */
    IF first.state THEN DO;
        new_cases = cases;
        new_deaths = deaths;
    END;
RUN;

/* Summary Statistics After Data Cleaning */
PROC MEANS DATA=us_covid_daily N MIN MAX MEAN STD;
    VAR new_cases new_deaths;
    TITLE "Summary Statistics After Data Cleaning";
RUN;

/* Step 7: Compute monthly cumulative new cases, deaths, and monthly death rate */
PROC SQL;
    CREATE TABLE covid_monthly_avg AS
    SELECT state, 
           MONTH(date) AS month, 
           YEAR(date) AS year,
           SUM(new_cases) AS total_new_cases,
           SUM(new_deaths) AS total_new_deaths,
           SUM(new_deaths) / SUM(new_cases) AS monthly_death_rate
    FROM us_covid_daily
    WHERE new_cases > 0 /* Avoid division by zero */
    GROUP BY state, year, month
    ORDER BY state, year, month;
QUIT;

/* Step 8: Visualization */
/* Line chart for monthly new cases */
PROC SGPLOT DATA=covid_monthly_avg;
    SERIES X=month Y=total_new_cases / GROUP=state;
    TITLE "COVID-19 Monthly New Cases for Each State";
    XAXIS LABEL='Month' GRID;
    YAXIS LABEL='New Cases' GRID;
RUN;

/* Line chart for monthly death rate */
PROC SGPLOT DATA=covid_monthly_avg;
    SERIES X=month Y=monthly_death_rate / GROUP=state;
    TITLE "COVID-19 Monthly Death Rate Trending for Each State";
    XAXIS LABEL='Month' GRID;
    YAXIS LABEL='Death Rate' GRID;
RUN;

/* Step 9: COVID-19 Monthly Summary Report */
PROC REPORT DATA=covid_monthly_avg NOWD;
    COLUMNS state year month total_new_cases total_new_deaths monthly_death_rate;
    DEFINE state / GROUP;
    DEFINE year / GROUP;
    DEFINE month / GROUP;
    DEFINE total_new_cases / ANALYSIS SUM "Total Cases";
    DEFINE total_new_deaths / ANALYSIS SUM "Total Deaths";
    DEFINE monthly_death_rate / COMPUTED FORMAT=percent8.2 "Death Rate";
    TITLE "COVID-19 Monthly Summary Report";
RUN;
