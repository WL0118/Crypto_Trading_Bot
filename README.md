# Crypto_Trading_Bot
** * Only for study, I recommend running this bot with the 'dryrun' mode on the Freqtrade system.*  **


This trading bot runs on the ['Freqtrade'](https://www.freqtrade.io/en/stable/) trading environment. The model used in the bot indicates critical price-increasing signals. 
It is made of LightGBM including statistical trading indicators as well as many manipulated features such as price difference between crypto markets.

We can check the trading results on the Telegram.

![image](https://user-images.githubusercontent.com/70621565/156699966-51949569-bbf6-40f3-9984-77ee93819156.png)


Also, we can check the signals and indicators on our cloud server.

![image](https://user-images.githubusercontent.com/70621565/156698599-09f07709-305e-44eb-a7ce-12c46fb978d9.png)

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

To find a relationship between past data and present data 'ShiftNadd' attaches a certain past row of selected features to the present row.

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



####  b. Set the target variable

I assumed that there are some special signals before a rapid price increase. Therefore, I put '1' in every row that their future price is a certain amount higher than their price. 

 *ex)Set criterium as **0.5** by 1 min*
||price|target|
|---|---|---|
|2022_01_01_00:00|100$|**1**(100.5-100 = 0.5)|
|2022_01_01_00:01|100.5$|**0**(100.4-100.5=-0.1)|
|2022_01_01_00:02|100.4$|**0**(100.5-100.4=0.1)|
|2022_01_01_00:03|100.5$|**1**(101.1-100.5 = 0.6)|
|2022_01_01_00:04|101$|Null|


## Freaqtrade

### 1. [Install Freqtrade(docker_quickstart)](https://www.freqtrade.io/en/stable/docker_quickstart/)

### 2. Docker-Compose 
Change the **'./docker/Dockerfile.custom'** File to import LightGBM on the docker environment. 

### 3. Strategy File

#### a. Get Candle realtime-data from the Binance and the Upbit.

#### b. Set realtime test dataframe
Using the candle data from step a, convert them into test data which has the same format as training data.

#### c. Signal.
When the model shows a signal, give the signal to the Freaktrade system to buy crypto-currency and automatically send a sell signal after certain minutes.

