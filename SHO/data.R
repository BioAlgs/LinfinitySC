# # Create empty data frames to store the stock price data
# stock_data_sho <- data.frame()
# stock_data_non_sho <- data.frame()

# # Define the date range
# start_date <- as.Date("2004-01-01")
# end_date <- as.Date("2005-12-31")

# stock_data_sho <- data.frame(matrix(ncol = 0, nrow = 504))
# stock_data_non_sho <- data.frame(matrix(ncol = 0, nrow = 504))


# # Loop through each symbol and download the stock price data
# for (i in 1:length(symbols)) {
#   symbol <- symbols[i]
#   tryCatch({
#     single.stock <- getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
#     single.stock <- data.frame(single.stock)
#     stock_data_sho[symbol] <- single.stock$AMD.Adjusted
#     if (sho_pilot[i] == 1) {
#       stock_data_sho[symbol] <- single.stock[,ncol(single.stock)]
#     } else {
#       stock_data_non_sho[symbol] <- single.stock[,ncol(single.stock)]
#     }
#     print(paste("Downloaded data for:", symbol))
#   }, error = function(e) {
#     print(paste("Error downloading data for:", symbol))
#   })
# }


# # Display the first few rows of each data frame
# head(stock_data_sho)
# head(stock_data_non_sho)



# install.packages("tidyquant")
# library(tidyquant)
# a <- tq_get("STTX")



# install.packages("yfR")
# library(yfR)
# tickers <- c("AAPL", "MSFT", "GOOG")
# data <- BatchGetSymbols(tickers = tickers, first.date = "2020-01-01", last.date = "2021-01-01")

# install.packages("alphavantager")
# library(alphavantager)
# # Set your API key
# av_api_key("2GJQ5DLGGO35NB72")

# # Example: Get daily stock data for Apple (AAPL)
# data <- av_get(symbol = "AAPL", av_fun = "TIME_SERIES_DAILY")
# # View the data
# head(data)

# Load necessary library
library(tidyverse)
library(readxl)
library(haven)
# # Read the local txt file using the pipeline delimiter
# stock.name <- read_excel('~/Git/Synth/SHO/one_drive/2005_Financial_Analysis_Sample.xlsx')
# sho_pilot <- stock.name$sho_pilot


## stata data
# CRSP_Daily_AprilMay2005 <- read_dta("Git/Synth/SHO/one_drive/GMW_LLS_codes/CRSP_Daily_AprilMay2005.dta")

# df_price <- CRSP_Daily_AprilMay2005 %>%
#   filter(tsymbol %in% stock.name$tsymbol) %>%
#   distinct(date, tsymbol, .keep_all = TRUE) %>%
#   select(date, tsymbol, prc) %>%
#   pivot_wider(names_from = tsymbol, values_from = prc)

# Read the Excel file
stock_name <- read_excel('~/Git/Synth/SHO/one_drive/2005_Financial_Analysis_Sample.xlsx')

# Read the Stata data
crsp_daily <- read_dta("Git/Synth/SHO/one_drive/GMW_LLS_codes/CRSP_Daily_AprilMay2005.dta")

# Filter, select, and pivot the CRSP data
df_price <- crsp_daily %>%
  filter(tsymbol %in% stock_name$tsymbol) %>%
  select(date, tsymbol, prc) %>%
  distinct(date, tsymbol, .keep_all = TRUE) %>%
  pivot_wider(names_from = tsymbol, values_from = prc)

# Prepare sho_pilot information
df_sho_pilot <- stock_name %>%
  select(tsymbol, sho_pilot) %>%
  distinct()

# Final selection if needed
# List of technology company tickers
tech_tickers <- c("PLXS", "SUNW", "ORCL", "MSFT", "EMC", "LLTC", "CY", "SGI", "CERN", 
                  "CMVT", "DELL", "MXIM", "BMC", "IBM", "AMAT", "TXN", "CSCO", 
                  "INTC", "QCOM", "NVLS", "MCHP", "MENT", "SNPS", "XLNX", "CRUS", 
                  "SYMC", "PMCS", "KLAC", "TER", "ATML", "ADI", "IDTI", "ZBRA", 
                  "ALTR", "ERTS", "INTU", "ADBE", "AAPL")
# Filter for tech companies in group 1 and group 0
tech_companies_group1 <- stock_name %>%
  filter(sho_pilot == 1 & tsymbol %in% tech_tickers) %>%
  pull(tsymbol)

tech_companies_group0 <- stock_name %>%
  filter(sho_pilot == 0 & tsymbol %in% tech_tickers) %>%
  pull(tsymbol)

df_select <- df_price %>%
  select(date, all_of(tech_tickers))

# Print the final data frame
head(df_select)
write.csv(df_select, file = "~/Desktop/Reg_SHO.csv", row.names = FALSE)
# write.csv(df_sho_pilot, file = "~/Desktop/Reg_SHO_only.csv", row.names = FALSE)

