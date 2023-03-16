library(utils)
library(tidyverse)
library(dplyr)
library(plyr)






# Dataset


## Canadian Gas
## freq = 'months'
canadianGas <- function() {

    destfile <- tempfile()
    download.file("https://github.com/robjhyndman/fpp3package/blob/master/data/canadian_gas.rda?raw=true",destfile)
    load(file=destfile)
    df <- canadian_gas
    df$ds <- df$Month
    df$y <- df$Volume
    tb <- df |> dplyr::select( ds, y) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))
    df <- data.frame(tb)

    return(tb)
}




# Bank Calls
# freq = 'days'
bankCalls <- function() {
    destfile <- tempfile()
    download.file("https://github.com/robjhyndman/fpp3package/blob/master/data/bank_calls.rda?raw=true",destfile)
    load(file=destfile)
    df <- as.data.frame(bank_calls)
    df <- df |> timetk::summarise_by_time(.date_var = DateTime,.by = "day",value = sum(Calls))
    #df$ds <- as_datetime(df$Month)
    df$ds <- as.Date(df$DateTime)
    df$y <- df$value

    tb <- df %>%
        select(y, ds) %>%
        as_tsibble(index = ds)

    calls_gaps <- tb %>% count_gaps(.full = TRUE)

    #tb <- fill_gaps(tb, .full = TRUE)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = 0L)

    return(tb)
}






## Insurance
# freq = 'months'
insurance <- function() {

    destfile <- tempfile()
    download.file("https://github.com/robjhyndman/fpp3package/blob/master/data/insurance.rda?raw=true",destfile)
    load(file=destfile)
    tb <- insurance
    tb$ds <- tb$Month
    tb$y <- tb$Quotes
    tb <- update_tsibble(tb, index = ds) |> dplyr::select(ds, y)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))

    return(tb)
}






## us_employment
# freq = 'months'
canadianGas <- function() {

    destfile <- tempfile()
    download.file("https://github.com/robjhyndman/fpp3package/blob/master/data/us_employment.rda?raw=true",destfile)
    load(file=destfile)
    df <- us_employment %>% filter(Title == 'Information')
    #df$ds <- as_datetime(df$Month)
    df$ds <- df$Month
    df$y <- df$Employed
    df <- df %>% update_tsibble(index = ds)
    df <- df %>% select(ds,y)

    tb <- df %>%
        select(y, ds) %>%
        as_tsibble(index = ds)

    gas_gaps <- tb %>% count_gaps(.full = TRUE)

    tb <- fill_gaps(tb, .full = TRUE)

    return(tb)
}




## Air Quality
# freq = 'hours'
airQuality <- function() {

    destfile <- tempfile()
    download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",destfile)
    df <- read.csv(unz(destfile, "AirQualityUCI.csv"), sep = ";" ,header = TRUE)
    df$ds <- lubridate::as_datetime(as.POSIXct(paste(df$Date, df$Time), format = "%d/%m/%Y%H.%M.%S"))
    df$y <- as.numeric(gsub(",", ".", df$CO.GT.))
    df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::select(ds,y) |> dplyr::distinct(ds,y)
    df <- df |> dplyr::filter(!is.na(ds)) |> dplyr::filter(ds > as.Date("2005-01-01"))
    df <- df |> dplyr::filter(!y < 0)
    df$ds <- lubridate::ymd_hms(df$ds)
    tb <- df |> dplyr::select(ds, y) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))
    df <- data.frame(tb)

    return(tb)
}






## SP500
# freq = 'months'
canadianGas <- function() {

    destfile <- tempfile()
    download.file("https://datahub.io/core/s-and-p-500/r/data.csv",destfile)
    df <-  readr::read_csv(destfile, show_col_types = FALSE)
    df$ds <- df$Date
    df$ds <- tsibble::yearmonth(df$ds)
    df$y <- df$SP500
    df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
    df <- df |> dplyr::filter(!is.na(ds))
    tb <- df |> dplyr::select(ds,y) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))
    df <- data.frame(tb)

    return(tb)
}







## Energy
energy <- function() {

    destfile <- tempfile()
    download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",destfile)
    df <-  readr::read_csv(destfile, show_col_types = FALSE)
    df$ds <- lubridate::ymd_hms(df$date)
    df$y <- df$Appliances

    df <- df |> dplyr::filter(ds > as.Date("2016-04-01"))
    df <- df |> tidyr::drop_na(c(ds,y))
    df <- df |> timetk::summarise_by_time(.date_var = ds,.by = "hour",y = sum(y))

    tb <- df |> dplyr::select(ds, y) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))

    return(tb)
}







## mars data
# freq = 'days'
marsData <- function() {

    destfile <- tempfile()
    download.file("https://raw.githubusercontent.com/the-pudding/data/master/mars-weather/mars-weather.csv",destfile)
    df <-  readr::read_csv(destfile, show_col_types = FALSE)
    df$ds <- lubridate::ymd(df$terrestrial_date)
    df$y <- df$max_temp
    df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
    df <- df |> dplyr::filter(!is.na(ds))
    tb <- df |> dplyr::select(y, ds) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))
    df <- data.frame(tb)

    return(tb)
}






## tesla
tesla <- function() {

    destfile <- tempfile()
    download.file("https://raw.githubusercontent.com/plotly/datasets/master/tesla-stock-price.csv",destfile)
    df <-  readr::read_csv(destfile, show_col_types = FALSE)
    df <- df[-c(1),]
    df$ds <- as.POSIXct(df$date, format = "%Y/%m/%d")
    df$ds <- lubridate::ymd(df$ds)
    df$y <- df$close
    df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
    df <- df |> dplyr::filter(!is.na(ds))
    tb <- df |> dplyr::select(y, ds) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))
    df <- data.frame(tb)

    return(tb)
}






## traffic
# freq = 'hours'
traffic <- function() {

    destfile <- tempfile()
    download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",destfile)
    df <- readr::read_csv(destfile, show_col_types = FALSE)
    df$ds <- lubridate::ymd_hms(df$date_time)
    df$y <- df$traffic_volume
    df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
    df <- df |> dplyr::filter(!is.na(ds)) |> dplyr::filter(ds > as.Date("2018-08-01"))
    tb <- df |> dplyr::select(y, ds) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))
    df <- data.frame(tb)

    return(tb)
}








## Sales
# freq = 'hours'
sales <- function() {

    destfile <- tempfile()
    download.file('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx',destfile)
    df <- readxl::read_excel(destfile)
    df <- df |> timetk::summarise_by_time(.date_var = InvoiceDate,.by = "hour",value = sum(Quantity*UnitPrice))
    df$ds <- lubridate::ymd_hms(df$InvoiceDate)
    df$y <- df$value
    df <- df |> dplyr::filter(!y < 0)
    df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
    df <- df |> dplyr::filter(!is.na(ds))
    df <- df |> dplyr::filter(ds > as.Date("2011-11-01"))
    tb <- df |> dplyr::select(ds,y) |> tsibble::as_tsibble(index = ds)
    tb <- tsibble::fill_gaps(tb, .full = TRUE, y = 0L)

    return(tb)
}






## flight data
# freq = 'months'
flights <- function() {

	destfile <- tempfile()
	download.file('https://download2.exploratory.io/downloads/data/2005_2006_flights.csv',destfile)
	df <- readr::read_csv(destfile, show_col_types = FALSE)
	download.file('https://download2.exploratory.io/downloads/data/2007_flights.csv',destfile)
	df1 <- readr::read_csv(destfile, show_col_types = FALSE)
	df <- rbind(df,df1)
	df$ds <- yearmonth(df$year_month)
	df$y <- df$count

	tb <- df |> dplyr::select(ds,y) |> tsibble::as_tsibble(index = ds)
	tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))

	return(tb)
}



## sp500
# freq = 'months'
sp500 <- function() {

	destfile <- tempfile()
	download.file("https://datahub.io/core/s-and-p-500/r/data.csv",destfile)
	df <-  readr::read_csv(destfile)
	df$ds <- df$Date
	df$ds <- tsibble::yearmonth(df$ds)
	df$y <- df$SP500
	df <- df %>% dplyr::filter(ds > as.Date("2005-01-01"))
	#df <- mutate(df, y = difference(y, order_by = ds, lag = 2, default = 0))

	df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
	df <- df |> dplyr::filter(!is.na(ds))

	tb <- df |> dplyr::select(y, ds) |> tsibble::as_tsibble(index = ds)
	tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))

	return(tb)

}





## Price Inflation UK
# freq = 'months'
inflationUK <- function() {

	destfile <- tempfile()
	fileName <- "https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindices/current/previous/v90/mm23.csv"
	download.file(url = fileName ,destfile = destfile, skip_empty_rows = TRUE)
	df <-  readr::read_csv(destfile)
	df <- df %>% select(Title,'CPI INDEX 00: ALL ITEMS 2015=100')
	names(df) <- c('ds','y')
	df <- df[1000:1500,]
	df$ds <- yearmonth(ym(df$ds))
	df <- df %>% tidyr::drop_na(c(ds,y))
	df$y <- as.numeric(df$y)

	#df <- mutate(df, y = difference(y, order_by = ds, lag = 2, default = 0))

	tb <- df |> dplyr::select(ds, y) |> tsibble::as_tsibble(index = ds)
	tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))

	return(tb)
}





## Covid Data
# freq = 'days'
covidStats <- function() {

	destfile <- tempfile()
	download.file("https://github.com/dsimband/rts/raw/main/data/population_by_country_2020.csv",destfile)
	pop_df <-  readr::read_csv(destfile)
	pop_df$CountryMap <- pop_df$"Country (or dependency)"
	pop_df[pop_df$CountryMap == "United States",]$CountryMap = "US"
	pop_df[pop_df$CountryMap == "China",]$CountryMap = "Mainland China"
	pop_df[pop_df$CountryMap == "United Kingdom",]$CountryMap = "UK"
	pop_df <- pop_df %>% dplyr::select(CountryMap,"Population (2020)")
	names(pop_df) <- c("Country","Population")



	destfile <- tempfile()
	download.file("https://github.com/dsimband/rts/raw/main/data/covid_19_data.csv",destfile)
	df <-  readr::read_csv(destfile)

	df$ds <- as.POSIXct(df$ObservationDate, format = "%m/%d/%Y")
	df$ds <- lubridate::ymd(df$ds)
	df$Country <- df$'Country/Region'

	countryNames <- df %>% dplyr::group_by(Country) %>%
		dplyr::summarise(
			Deaths = sum(Deaths),
			Confirmed = sum(Confirmed),
			Recovered = sum(Recovered),
		) %>%
		dplyr::arrange(desc(Confirmed), n=10) %>%
		dplyr::select(Country) %>% head(10)

	#countryNames <- countryNames$Country
	countryNames <- c('US')

	df <- df %>% dplyr::group_by(ds,Country) %>%
		dplyr::filter(Country %in% countryNames) %>%
		dplyr::summarise(
			Deaths = sum(Deaths),
			Confirmed = sum(Confirmed),
			Recovered = sum(Recovered),
		) %>%
		dplyr::select(ds, Country, Deaths, Confirmed, Recovered)

	df <- merge(df, pop_df, by.x = "Country", by.y = "Country")
	df <- df %>% dplyr::mutate(
							Active = Confirmed - Recovered -Deaths,
							Active_percent = Active / Population,
							y = Active_percent)

	#df <- df |> tidyr::drop_na(c(ds,y)) |> dplyr::distinct(ds,y)
	df <- df |> dplyr::filter(!is.na(ds))

	tb <- df |> dplyr::select(ds, Country,y, Active, Confirmed, Recovered, Deaths) |>
		tsibble::as_tsibble(index = ds)
	tb <- tsibble::fill_gaps(tb, .full = TRUE, y = dplyr::last(y))

	return(tb)


}






## Covid Data
# freq = 'days'
covidDeaths <- function() {


library(mosaicData)




	destfile <- tempfile()
	download.file("https://zenodo.org/record/5129091/files/temperature_rain_dataset_without_missing_values.zip?download=1",destfile)
	#df <- readr::read_tsv(destfile)
	df <-  readr::read_csv(destfile, skip=15, col_names=FALSE)


	df <- mosaicData::Births

	df <- mosaicData::Weather %>% dplyr::filter(city =='Chicago')
	df <- mosaicData::Weather %>% dplyr::filter(city =='San Diego')



	library(tsdl)
	#t <- subset(tsdl,"Sport").as_tibble()
	t <- subset(tsdl,"Sport")
	t[[2]]
	t <- as_tsibble(t[[2]])

	m <- subset(tsdl,"Macroeconomic")
	m[4]
	m[[4]]



	library(Mcomp)
	d <- subset(M1, "INDUST", "monthly")

}

