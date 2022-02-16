# Crypto_Trading_Bot
** * Only for study, I recommend running this bot with the 'dyrun' mode on the Freqtrade system.*  **


This trading bot runs on the ['Freqtrade'](https://www.freqtrade.io/en/stable/) as a strategy. The model used in the bot indicates critical price-increasing signals. 
It is made of LightGBM including statistical trading indicators as well as many manipulated features such as price difference between crypto markets.



## The Model

### 1. Dowload Datasets.
This model uses data sets from two different crypto markets to extract features related to the a same crypto-currency price difference between from two markets.
I got 1-min candle data from the Binance market and 1-day candle data from the Upbit market by using each Rest API.(You can find it in the Kaggle-Notebook)

### 2. Training set

####  a. Generate some features and statistical indicatorors 

##### ※ [*MACD*](https://en.wikipedia.org/wiki/MACD) ※

##### ※ [*Bolinger Band*](https://en.wikipedia.org/wiki/Bollinger_Bandshttps://en.wikipedia.org/wiki/Bollinger_Bands) ※

##### ※ [*RSI*](https://en.wikipedia.org/wiki/Relative_strength_index) ※

##### ※ [*MFI*](https://en.wikipedia.org/wiki/Money_flow_index) ※

##### ※ Price difference ※
Match and set time index both the Upbit data and the Binance data and calculate the crypto price difference between the Binance and the Upbit(Korean)

##### ※ *AmOfChanges* ※

Calculate changes for selected features in a certain time period.

##### ※ *ShiftNadd* ※



||price|RSI|AmOfChanges_1|
|---|---|---|---|
|2022_01_01|1$|30|Null|
|2022_01_02|2$|35|1|
|2022_01_03|3$|40|0.3|


Apply the 1 ShiftNadd with 'Price', 'RSI' and 'AmOfChanges_1' 

||price|RSI|AmOfChanges_1|price_SnA_1|RSI_SnA_1|AmOfChanges_1_SnA_1|
|---|---|---|---|---|---|---|
|2022_01_01|&#x1F538;1$|&#x1F538;30|&#x1F538;Null|Null|Null|Null|
|2022_01_02|&#x1F537;**2$**|&#x1F537;**35**|&#x1F537;**1**|&#x1F538;1$|&#x1F538;30|&#x1F538;Null|
|2022_01_03|3$|40|0.3|&#x1F537;**2$**|&#x1F537;**35**|&#x1F537;**1**|


To find a relationship between past data and present data 'ShiftNadd' attaches a certain past row of selected features to the present row.

####  b.
####  c.
####  d.

## Freaqtrade strategy 
