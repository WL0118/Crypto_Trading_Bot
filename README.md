# Crypto_Trading_Bot
** * Only for study, I recommend running this bot with the 'dyrun' mode on the Freqtrade system.*  **


This trading bot runs on the 'Freqtrade' as a strategy. The model used in the bot indicates critical price-increasing signals. 
It is made of LightGBM including statistical trading indicators as well as many manipulated features such as price difference between crypto markets.



## The Model


### 1. Dowload Datasets.
This model uses data sets from two different crypto markets to extract features related to the a same crypto-currency price difference between from two markets.
I got 1-min candle data from the Binance market and 1-day candle data from the Upbit market by using each Rest API.(You can find it in the Kaggle-Notebook)

### 2. Training set
####  a. Calculate Price difference
Match and set time index both the Upbit data and the Binance data
Calculate the crypto price difference between the Binance and the Upbit(Korean)
####  b. Calculate statistical information and put them into each column 

##### *MACD*

##### *Bolinger Band*

##### *RSI*

##### *MFI_14*

##### *SLOPE*

##### *ShiftNadd*

####  c.
####  d.
####  e.

## Freaqtrade strategy 
