# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime as dt
# from modify_df import *
import pickle
import lightgbm as lgb
#pip install lightgbm
#apt-get install libgomp1
from lightgbm import LGBMClassifier
import joblib


import requests
import time
from datetime import datetime


import logging

logger = logging.getLogger(__name__)


# This class is a sample. Feel free to customize it.
class SampleStrategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Hyperoptable parameters
    #buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    #sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 250

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # plot_config = {
    #     'main_plot': {
    #         'tema': {},
    #         'sar': {'color': 'white'},
    #     },
    #     'subplots': {
    #         "MACD": {
    #             'macd': {'color': 'blue'},
    #             'macdsignal': {'color': 'orange'},
    #         },
    #         "RSI": {
    #             'rsi': {'color': 'red'},
    #         }
    #     }
    # }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        TrainStartYear = 2021
        TrainStartMonth = 1
        TrainStartDay =1

        start_train_time = datetime(int(TrainStartYear),int(TrainStartMonth),int(TrainStartDay)).replace(tzinfo= dt.timezone.utc)
        #crypto
        crypto_=metadata['pair'].split('/')[0]
        req_DAY_len = 250
        
        Slope_1_min=['Open', 
               'VolumeMean_5', 'Volumeunit_STD_5', 
               'VolumeMean_20', 'Volumeunit_STD_20', 
                'CloseMean_5', 'Closeunit_STD_5',
                'CloseMean_20', 'Closeunit_STD_20',
                'CloseMean_60', 'Closeunit_STD_60',
                'CloseMean_120', 
                'CloseMean_200', 'MACD','RSI_14',
               'slow_K_1', 'slow_D_1',
                    ]
        Slope_3_5_min=[
               'VolumeMean_5', 
               'VolumeMean_20',
                'CloseMean_5', 
                 'CloseMean_20', 
                'CloseMean_60',
                'CloseMean_120',
                'CloseMean_200', 
                 'MACD' ,'RSI_14']
        max_min = ['CloseMax_20','CloseMax_60',
                  'CloseMin_20','CloseMin_60','CloseMin_200'
                  ]


        
        if(dataframe['date'].iloc[-1].hour==0 and dataframe['date'].iloc[-1].minute==0):
            binanace_usdt_crypto = crypto_+"USDT"
            upbit_usdt_crypto = "USDT-"+crypto_
            DayDataFrame().binance2csv(binanace_usdt_crypto,req_DAY_len)

            
            DayDataFrame().upbit2csv(Coin_Name=upbit_usdt_crypto, candleFrequent='D', req_number=req_DAY_len,Last_time ='')

            
            modify_df().modify_day_df(crypto_,req_DAY_len)






        Day_df_name = "/freqtrade/user_data/data/DAY/"+"DAY_"+crypto_+"_"+str(req_DAY_len)+".csv"
        Day_df = pd.read_csv(Day_df_name)
        dataframe.rename(columns={'open':'Open',   'high':'High',      'low':'Low',    'close':'Close',    'volume':'Volume'}, inplace=True)
        
        dataframe['DAY_UTC_Day'] = dataframe.apply(lambda x : str(x['date'].year)+'-'+str(x['date'].month)+'-'+str(x['date'].day),axis=1)
        dataframe = dataframe.merge(Day_df,on='DAY_UTC_Day')
        
        dataframe['Min_delta']=dataframe.apply(lambda x : (x['date'] - start_train_time).total_seconds()/60,axis=1)
        dataframe = modify_df().BB(dataframe ,'Volume' , BB_len=5, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Volume' , BB_len=20, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Volume' , BB_len=60, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Volume' , BB_len=120, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Volume' , BB_len=200, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Close' , BB_len=5, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Close' , BB_len=20, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Close' , BB_len=60, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Close' , BB_len=120, unit=2,isFutureVal=0)
        dataframe = modify_df().BB(dataframe ,'Close' , BB_len=200, unit=2,isFutureVal=0)  
        

        
        ########################################RSI ##################################################################
        #df = RSI(df,isFutureVal=0)
        dataframe['RSI_14'] = ta.RSI(dataframe['Close'],timeperiod = 14)
        ##########################################################################################################

        ##########################################################################################################
        #df = MACD(df,isFutureVal=0)
        macd,signal,_=ta.MACD(dataframe['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['MACD'] = macd
        dataframe['MACD_signal'] = signal

        ##########################################################################################################
        #df = MFI(df,isFutureVal=0)
        dataframe['MFI_14'] = ta.MFI(dataframe['High'], dataframe['Low'], dataframe['Close'], dataframe['Volume'], timeperiod=14)
        
        ##############################################################################################################
        #df = Stochastic(df,isFutureVal=0)
        dataframe['slow_K_1'], dataframe['slow_D_1'] = ta.STOCH(dataframe['High'], dataframe['Low'], dataframe['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        ################################################################################################################
        dataframe = modify_df().Slope(dataframe,1,Slope_1_min)
        dataframe = modify_df().Slope(dataframe,3,Slope_3_5_min)
        dataframe = modify_df().Slope(dataframe,5,Slope_3_5_min)
        dataframe = dataframe.replace(np.inf, np.nan)
        dataframe = modify_df().ShiftNadd(dataframe,max_min,1)
        dataframe = modify_df().ShiftNadd(dataframe,max_min,2)
        dataframe = modify_df().ShiftNadd(dataframe,max_min,3)
        dataframe = modify_df().ShiftNadd(dataframe,[ele+'_slope_1' for ele in Slope_1_min],1)
        dataframe = modify_df().ShiftNadd(dataframe,[ele+'_slope_1' for ele in Slope_1_min],2)
        dataframe = modify_df().ShiftNadd(dataframe,[ele+'_slope_1' for ele in Slope_1_min],3)
        dataframe['DAY_Kimchi_PriMean_5'] = dataframe.apply(lambda x :np.nan if np.isnan(x['DAY_Kimchi_PriMean_5']) else round(x['DAY_Kimchi_PriMean_5']),axis=1)
        
        dataframe = dataframe.drop(columns=['DAY_UTC_Day'])

        MODEL_PATH = '/freqtrade/user_data/data/models/'
        MODEL_NAME = 'ETH_1_Binance_2021_1_3.pkl'
        # warning in here        
        with open(MODEL_PATH+MODEL_NAME, 'rb') as f:
            model = joblib.load(f)

        dataframe['buy_330'] = model[0].predict(dataframe.iloc[:,1:])
        dataframe['buy_3_roll_sum'] = dataframe['buy_330'].rolling(3).sum()
        dataframe['buy_2_roll_sum'] = dataframe['buy_330'].rolling(2).sum()
        dataframe['sell_330'] = dataframe.apply(lambda x: 1 if(x['buy_3_roll_sum']>0 and x['buy_2_roll_sum']==0 ) else 0, axis=1)
        
        
        dataframe.drop(dataframe.columns[6:-4], axis=1, inplace=True)
        dataframe.drop(columns=['buy_3_roll_sum','buy_2_roll_sum'], axis=1, inplace=True)
        dataframe.rename(columns={'Open':'open',   'High':'high',      'Low':'low',    'Close':'close',    'Volume':'volume'}, inplace=True)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['buy_330']==1)&
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                #(qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &
                (dataframe['sell_330']==1)&
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
class DayDataFrame():
    def send_request(self, reqType, reqUrl, reqParam, reqHeader):
        try:

            # 
            err_sleep_time = 0.3

            while True:

                response = requests.request(reqType, reqUrl, params=reqParam, headers=reqHeader)

                if 'Remaining-Req' in response.headers:

                    hearder_info = response.headers['Remaining-Req']
                    start_idx = hearder_info.find("sec=")
                    end_idx = len(hearder_info)
                    remain_sec = hearder_info[int(start_idx):int(end_idx)].replace('sec=', '')
                else:
                    break
                if int(remain_sec) < 3:
                    time.sleep(err_sleep_time)
                if response.status_code == 200 or response.status_code == 201:
                    break
                elif response.status_code == 429:
                    time.sleep(err_sleep_time)
                else:

                    break


            return response

        except Exception:
            raise

    def get_candle(self,target_item, tick_kind, inq_range, Last_time):
        try:

            if tick_kind == "1" or tick_kind == "3" or tick_kind == "5" or tick_kind == "10" or tick_kind == "15" or tick_kind == "30" or tick_kind == "60" or tick_kind == "240":
                target_url = "minutes/" + tick_kind
            # Day
            elif tick_kind == "D":
                target_url = "days"
            # Week
            elif tick_kind == "W":
                target_url = "weeks"
            # Monthget_candle
            elif tick_kind == "M":
                target_url = "months"
            # fail
            else:
                raise Exception("fail:" + str(tick_kind))

            querystring = {"market": target_item, "count": inq_range, "to":Last_time}
            res = self.send_request("GET", "https://api.upbit.com" + "/v1/candles/" + target_url, querystring, "")
            candle_data = res.json()


            return candle_data
        except Exception:
            raise
            

        
    def upbit2csv(self,Coin_Name,candleFrequent,req_number,Last_time =''):
        #Last_time = '2021-9-18'+'T'+'18:15:01'+'Z'
        roop_200 = req_number//200
        rest_200 = req_number%200
        old_data= self.get_candle(Coin_Name, candleFrequent, str(rest_200),Last_time)
        if roop_200>=1:
            for num in range(roop_200):
                if num%100==0:

                    new_time = old_data[-1]["candle_date_time_utc"]+'Z'
                try:
                    new_data= self.get_candle(Coin_Name, candleFrequent, "200",new_time)
                except:
                    break
                old_data =  old_data + new_data
        newdata=[]
        for i in old_data:
            OHLC={}
            OHLC["KST"] = i['candle_date_time_kst']
            OHLC["UTC"] = i['candle_date_time_utc']
            OHLC["opening_price"] = i['opening_price']
            OHLC["high_price"] = i['high_price']
            OHLC["low_price"] = i['low_price']
            OHLC["trade_price"] = i['trade_price']
            OHLC["acc_trade_price"] = i['candle_acc_trade_price']
            OHLC["acc_trade_volume"] = i ['candle_acc_trade_volume']
            newdata.append(OHLC)
        df=pd.DataFrame(newdata)

        start_time = df["UTC"][len(df)-1].replace(':','-')
        end_time = df["UTC"][0].replace(':','-')
        df_name = "/freqtrade/user_data/data/DAY/upbit/"+Coin_Name+"_"+candleFrequent+'_'+str(req_number)+"_upbit.csv"
        df.to_csv(df_name,index=False,encoding="cp949")
        
    def binance2csv(self,Coin_Name_bi,req_number_bi,candleFrequent_bi='1d'):
        
        COLUMNS = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades', 
                   'tb_base_av', 'tb_quote_av', 'ignore']
        
        params = {
                'symbol': Coin_Name_bi,
                'interval': candleFrequent_bi,
                'limit': req_number_bi
            }
        
        result = requests.get('https://api.binance.com/api/v3/klines', params = params)
        
        js = result.json()
        df = pd.DataFrame(js)
        df.columns = COLUMNS
        df['Open_time'] = df.apply(lambda x:datetime.fromtimestamp(x['Open_time']//1000 ), axis=1)
        
        df = df.drop(columns = ['Close_time', 'ignore'])
        df['Symbol'] = Coin_Name_bi
        df.loc[:, 'Open':'tb_quote_av'] = df.loc[:, 'Open':'tb_quote_av'].astype(float)  # string to float
        df['trades'] = df['trades'].astype(int)

        df_name = "/freqtrade/user_data/data/DAY/binance/"+Coin_Name_bi+"_"+candleFrequent_bi+'_'+str(req_number_bi)+"_binance.csv"

        df.to_csv(df_name,index=False,encoding="cp949")




class modify_df():
    def modify_day_df(self,cryptoName,lenOfDay):
        DAY_Slope_1=['opening_price', 'Volume',
               'Kimchi_Pri', 'Kimchi_PriMean_5', 'Kimchi_Priunit_STD_5', 'Kimchi_Priunit_STD_20',
               'VolumeMean_5', 'Volumeunit_STD_5', 
               'VolumeMean_20', 'Volumeunit_STD_20', 
                'CloseMean_5', 'Closeunit_STD_5',
                'CloseMean_20', 'Closeunit_STD_20',
                'MFI_14', 'MACD','RSI_14',
               'slow_K_1', 'slow_D_1']
        DAY_Slope_3_5=[
               'Kimchi_PriMean_5', 'Kimchi_Priunit_STD_5','Kimchi_PriMean_20', 'Kimchi_Priunit_STD_20',
               'VolumeMean_5', 
               'VolumeMean_20',
                'CloseMean_5', 
             'CloseMean_20', 
                 'RSI_14',
                'MFI_14']    
        if lenOfDay <200:
            print("check: lenOfDay")
            return
        
        crypto_bi = cryptoName+"USDT"
        crypto_up = "USDT-"+cryptoName

        binanceCrytoFileNameDIR= "/freqtrade/user_data/data/DAY/binance/"+crypto_bi+"_1d_"+str(lenOfDay)+"_binance.csv"
        upbitCrytoFileNameDIR = "/freqtrade/user_data/data/DAY/upbit/"+crypto_up+"_D_"+str(lenOfDay)+"_upbit.csv"
        binanceDay_df= pd.read_csv(binanceCrytoFileNameDIR)
        upbitDay_df= pd.read_csv(upbitCrytoFileNameDIR)



        upbitDay_df['UTC'] = upbitDay_df.apply(lambda x : x['UTC'][:10]+' '+x['UTC'][11:19], axis = 1 )
        binanceDay_df['Open_time'] = binanceDay_df.apply(lambda x : x['Open_time'][:10]+" 00:00:00", axis = 1 )

    #     #merge two df
        upbitDay_df = pd.merge(binanceDay_df[['Open_time','Open']],upbitDay_df, left_on='Open_time',right_on='UTC')
        #calculate price difference between korean crypto price and american cryto price
        upbitDay_df['Kimchi_Pri'] = upbitDay_df.apply(lambda x : (x['opening_price']/(x['Open']))*100 if ~np.isnan(x['Open']) else np.nan,axis=1)
    
        upbitDay_df['UTC_Day'] = upbitDay_df.apply(lambda x : x['UTC'][:4]+'-'+str(int(x['UTC'][5:7]))+'-'+str(int(x['UTC'][8:10])),axis=1)
        
        upbitDay_df['Kimchi_Pri']=round(upbitDay_df['Kimchi_Pri'])

    #     #change column names
        upbitDay_df.rename(columns={'high_price': 'High', 'low_price': 'Low', 'trade_price': 'Close'}, inplace=True)
        upbitDay_df.rename(columns={'acc_trade_volume': 'Volume'}, inplace=True)

    #calculte statistical information and put it inot each column 
        upbitDay_df = self.BB(upbitDay_df ,'Kimchi_Pri' , BB_len=5, unit=2,isFutureVal=0)
        upbitDay_df = self.BB(upbitDay_df ,'Kimchi_Pri' , BB_len=20, unit=2,isFutureVal=0)
        upbitDay_df = self.BB(upbitDay_df ,'Volume' , BB_len=5, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Volume' , BB_len=20, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Volume' , BB_len=60, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Volume' , BB_len=120, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Volume' , BB_len=200, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Close' , BB_len=5, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Close' , BB_len=20, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Close' , BB_len=60, unit=2)
        upbitDay_df = self.BB(upbitDay_df ,'Close' , BB_len=120, unit=2)

        ########################################RSI ##################################################################
        #upbitDay_df = RSI(upbitDay_df)
        upbitDay_df['RSI_14'] = ta.RSI(upbitDay_df['Close'],timeperiod = 14)
        upbitDay_df[['RSI_14']] = upbitDay_df[['RSI_14']].shift(1)
        ##########################################################################################################

        ##########################################################################################################
        #upbitDay_df = MACD(upbitDay_df)
        macd,signal,_=ta.MACD(upbitDay_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        upbitDay_df['MACD'] = macd
        upbitDay_df['MACD_signal'] = signal
        upbitDay_df['MACD'] = upbitDay_df['MACD'].shift(1)
        upbitDay_df['MACD_signal'] = upbitDay_df['MACD_signal'].shift(1)
        ##########################################################################################################

        #upbitDay_df = MFI(upbitDay_df)
        upbitDay_df['MFI_14'] = ta.MFI(upbitDay_df['High'], upbitDay_df['Low'], upbitDay_df['Close'], upbitDay_df['Volume'], timeperiod=14)
        upbitDay_df['MFI_14'] = upbitDay_df['MFI_14'].shift(1)
        ##############################################################################################################

        #upbitDay_df = Stochastic(upbitDay_df)
        upbitDay_df['slow_K_1'], upbitDay_df['slow_D_1'] = ta.STOCH(upbitDay_df['High'], upbitDay_df['Low'], upbitDay_df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        upbitDay_df['slow_K_1'] = upbitDay_df['slow_K_1'].shift(1)
        upbitDay_df['slow_D_1'] = upbitDay_df['slow_D_1'].shift(1)
        ################################################################################################################

        upbitDay_df = self.Slope(upbitDay_df,1,DAY_Slope_1)
        upbitDay_df = self.Slope(upbitDay_df,3,DAY_Slope_3_5)
        upbitDay_df = self.Slope(upbitDay_df,5,DAY_Slope_3_5)
        upbitDay_df = upbitDay_df.replace(np.inf, np.nan)
        upbitDay_df = self.ShiftNadd(upbitDay_df,[ele+'_slope_1' for ele in DAY_Slope_1],1)
        upbitDay_df = self.ShiftNadd(upbitDay_df,[ele+'_slope_1' for ele in DAY_Slope_1],2)
        #put 'DAY' for all columns to distinguish Day data and min data


        upbitDay_df.columns = ['DAY_'+ele for ele in upbitDay_df.columns]
        upbitDay_df = upbitDay_df.drop(columns=['DAY_High', 'DAY_Low', 'DAY_Close', 'DAY_acc_trade_price','DAY_Open_time', 'DAY_Open', 'DAY_KST', 'DAY_UTC','DAY_Volume'])
        
        df_name = "/freqtrade/user_data/data/DAY/"+"DAY_"+cryptoName+"_"+str(lenOfDay)+".csv"
        upbitDay_df.to_csv(df_name,index=False,encoding="cp949")
    
    
    
    def Slope(self,df,shiftNum,shiftCols):
        if len(df)<abs(shiftNum):
            print("check_range")
            return

        df = df.copy()

        for col in shiftCols:
            df[col+'_slope_%d'%shiftNum] = df[col]/df[col].shift(shiftNum)

        return df


    def ShiftNadd(self,df,col_names,shift_num):
        df = df.copy()
        df[[col+'_shift_'+str(shift_num) for col in col_names]] = df[col_names].shift(shift_num)
        return df

    def BB(self,df ,Col_name , BB_len, unit=2,isFutureVal=1):
        if len(df)<BB_len:
            print("check_range")
            return
        df=df.copy()
        rolling = df[Col_name].rolling(BB_len)
        if isFutureVal:
            df[Col_name+'Mean_'+str(BB_len)] = rolling.mean().shift(1)
            df[Col_name+'unit_STD_'+str(BB_len)]=rolling.std().shift(1)*unit
            df[Col_name+'Min_'+str(BB_len)] = rolling.min().shift(1)
            df[Col_name+'Max_'+str(BB_len)]= rolling.max().shift(1)
        else:
            df[Col_name+'Mean_'+str(BB_len)] = rolling.mean()
            df[Col_name+'unit_STD_'+str(BB_len)]=rolling.std()*unit
            df[Col_name+'Min_'+str(BB_len)] = rolling.min()
            df[Col_name+'Max_'+str(BB_len)]= rolling.max()
        return df
