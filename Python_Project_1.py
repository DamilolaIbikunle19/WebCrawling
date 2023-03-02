# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 08:56:13 2022

@author: Damilola Ibikunle
"""

import requests
import pandas as pd
import sqlite3
import numpy as np
import math
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from bs4 import BeautifulSoup as BS
from bs4 import NavigableString
import statsmodels.api as sms
import statsmodels.formula.api as smf

## Question 1 - Download all the master index EDGAR Filings in 2021 ##

heads = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0"}

q1_file_name = 'master_index_2021QTR1.csv'
q1_url = r'https://www.sec.gov/Archives/edgar/full-index/2021/QTR1/master.idx'
q2_file_name = 'master_index_2021QTR2.csv'
q2_url = r'https://www.sec.gov/Archives/edgar/full-index/2021/QTR2/master.idx'
q3_file_name = 'master_index_2021QTR3.csv'
q3_url = r'https://www.sec.gov/Archives/edgar/full-index/2021/QTR3/master.idx'
q4_file_name = 'master_index_2021QTR4.csv'
q4_url = r'https://www.sec.gov/Archives/edgar/full-index/2021/QTR4/master.idx'

def get_edgar_quarterly_filing(url, file_name):
    ##
    # get_edgar_quarterly_filings- This is the get_edgar_quarterly_filings code
    #
    # Description: This function downloads the quarterly edgar filing data
    # @url: it takes in the URL of the quarterly data as parameter
    # @file_name: It takes the file name with which to save the data as parameter
    # Return: It doesn't return anything
    ##
    request_content = requests.get(url, headers=heads).content
    result = request_content.decode("utf-8", "ignore")
    with open(file_name,'w') as f:
        f.write(result)  

get_edgar_quarterly_filing(q1_url, q1_file_name)
get_edgar_quarterly_filing(q2_url, q2_file_name)
get_edgar_quarterly_filing(q3_url, q3_file_name)
get_edgar_quarterly_filing(q4_url, q4_file_name)

    
# Open each of the master index files and loop through it to download the filings in each quarter #

def download_filings (filename, url):
    ##
    # download_filings- This is the download_filings function
    #
    # Description: This function downloads the individual filings of individual companies
    # @url: it takes in the URL of the quarterly data as parameter
    # @file_name: It takes the file name with which to read the individual company URL from
    # Return: It doesn't return anything
    ##
    with open(filename,'r') as f:
        lines = f.readlines()

    records = [tuple(line.split('|')) for line in lines[11:]]
    for r in records:
        cik = r[0]
        form = r[2]
        date = r[3]
        filing = r'https://www.sec.gov/Archives/'+ r[4]
        request_content = requests.get(url[:-1], headers = heads).content
        result = request_content.decode("utf-8", "ignore")
        with open ('{0}_{1}_{2}_{3}.csv'.format(cik,form,date,filing).replace('/', ' '), 'w') as f:
            f.write(result)


download_filings(q1_file_name, q1_url)
download_filings(q2_file_name, q2_url)
download_filings(q3_file_name, q3_url)
download_filings(q4_file_name, q4_url)


# Question 6 - Conduct Textual Analysis of the sampled firm's 10-Q filings
def loughran_McDonald_Dictionary():
    ##
    # loughran_McDonald_Dictionary- This is the loughran_McDonald_Dictionary function
    #
    # Description: This function creates a a list of negative and positive words from the loughran dictionary
    # Return: This function returns 2 lists. A negative words and positive words list
    ##
    lexicon = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2021_simplified_ver.csv')
    mask = lexicon['negative'] == 1
    negatives = lexicon[mask]['Word']
    neg_words = negatives.tolist()
    mask = lexicon['positive'] == 1
    positives = lexicon[mask]['Word']
    pos_words = positives.tolist()
    return neg_words, pos_words


def url_to_file(file_url):
    ##
    # url_to_file- This is the url_to_file function
    #
    # Description: This function gets the content of the url fed to it
    # @file_url: it takes in the file url as a parameter
    # Return: It returns the content of the file_url
    ##
    stripped_url = file_url.strip()
    try:
        request_ = requests.get(stripped_url, headers=heads)
        request_content = request_.content
        results = request_content.decode("utf-8", "ignore")
    except Exception as e:
        print("Request encountered an error: ", e)
    return results


def get_words_in_file(url_file):
    ##
    # get_words_in_file- This is the get_words_in_file function
    #
    # Description: This function extracts the words in the url_file gotten from url_to_file
    # @url_file: it takes in the content of the file returned by the url_to_file function as parameter
    # Return: It returns a list of the words in the file
    ##
    try:
        soup = BS(url_file, 'html.parser')
        text = soup.findAll(text=lambda text: isinstance(text, NavigableString))
        text = u" ".join(text)
    except Exception as e:
        print("Beautiful soup encountered an error: ", e)
    return text.split()


def get_tone(words_in_file, neg_words, pos_words):
    ###
    # get_tone- This is the get_tone function
    #
    # Description: This function calculates the tone of the filings based on the words found in it
    # @words_in_file: it takes in the list of words in the file as gotten from the get_words_in_file function
    # @neg_words: It takes the list of negative words generated by the loughram_McDonald dictionary
    # @pos_words: It takes the list of positive words generated by the loughram_McDonald dictionary
    # Return: It returns the tone of the words found in the word list.
    # tone = (total number of positive words - total number of negative words) / (total number of positive words + total number of negative words + 1)
    ##
    neg_count = 0
    pos_count = 0
    for word in words_in_file:
        if word.upper() in neg_words:            
            neg_count += 1        
        if word.upper() in pos_words:
            pos_count += 1        
    return (pos_count - neg_count)/(pos_count + neg_count + 1)

negative_words, positive_words = loughran_McDonald_Dictionary()
      

## Open each of the master index files, carry out a textual analysis and store in a dataframe ##
Base_url = "https://www.sec.gov/Archives/"


def get_quarterly_filings(filename):
    ##
    # get_quarterly_filings- This is the get_quarterly_filings function
    #
    # Description: This function creates a dataframe of the quarterly filings data
    # @filename: it takes the quarterly data file name
    # Return: It returns a dataframe of the quarterly filing data.
    ##
    with open(filename,'r') as f:
        lines = f.readlines()
    records = [tuple(line.split('|')) for line in lines[11:]]
    database = []
    for r in records:
        tone = 0
        database.append ([r[0], r[2], r[3], r[4], tone])
    
    df = pd.DataFrame(database, columns = ['cik', 'form', 'date', 'filing', 'tone'])
    return df


def add_tone(p_dataframe, Base_url):
    ##
    # add_tone- This is the add_tone function
    #
    # Description: This function adds the tone to the dataframe
    # @p_dataframe: it takes a dataframe that needs tone added to it as parameter
    # Return: It returns a dataframe with the tone column populated
    ##
    p_dict = p_dataframe.to_dict('index')
    for index, values in p_dict.items():                
        if type(values['filing']) == str:
            try:
                url_file = url_to_file(Base_url + values['filing'])
                filing_words = get_words_in_file(url_file)
                tone = get_tone(filing_words, negative_words, positive_words)
                values['tone'] = tone
            except Exception as e:
                values['tone'] = "inaccessible"
                print("We encountered an error assessing the words in this file ", Base_url + values['filing'], " The error: ", e)

    project_df_new = pd.DataFrame(p_dict)
    project_df_new = project_df_new.transpose()
    return project_df_new


Q1_df = get_quarterly_filings(q1_file_name)
Q2_df = get_quarterly_filings(q2_file_name)
Q3_df = get_quarterly_filings(q3_file_name)
Q4_df = get_quarterly_filings(q4_file_name)

project_dataframe = pd.concat([Q1_df, Q2_df, Q3_df, Q4_df])
project_dataframe.reset_index(inplace=True, drop=True)

           
#Question 2 - Create a database to be imported to sql and discuss the database
#The discussion of the database was done on Sql

con = sqlite3.connect('Project_database.db')
cur = con.cursor()
command = "CREATE TABLE Project_table (cik text, form text, date date, filing, tone);"
cur.execute(command)

project_dataframe.to_sql('Project_table' , con, if_exists='replace', index = False)

con.commit()
con.close()


# Question 3 - Query 10-Q filings from the Database
con = sqlite3.connect('Project_database.db')
cur = con.cursor()
command = "SELECT * FROM Project_table"
result = cur.execute(command)
result.description 
cols = list(map(lambda x: x[0], result.description))
result_table = result.fetchall()
dataf = pd.DataFrame(result_table, columns=cols)


command = "SELECT * FROM Project_table WHERE form LIKE '10-Q';"
result = cur.execute(command)
cols = list(map(lambda x: x[0], result.description))
result_table = result.fetchall()
Queried_dataframe = pd.DataFrame(result_table, columns=cols)
Queried_dataframe = Queried_dataframe.astype({'cik':'int'})


#Question 4 - Download the stock prices of some companies that filed 10Q forms in 2021#
# This was used to get the EDGAR tickers for stock prices https://www.sec.gov/files/company_tickers.json#

yf.pdr_override()
tickers = ['^GSPC', 'AAPL', 'MSFT', 'NVDA', 'V', 'MA', 'AVGO', 'CSCO', 'ACN', 'ADBE', 'TXN', 'UNH', 'JNJ', 'LLY', 'ABBV', 'PFE', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'TGT', 'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'SCHW', 'MS', 'SPGI', 'BLK', 'AXP']
start_date = '2021-01-01'
end_date = '2021-12-31'

def get_stockprices(tickers):
    ##
    # get_stockprices- This is the get_stockprices function
    #
    # Description: This function gets the stockprices of 41 selected stocks and adds it to a dataframe
    # @stockprices_dataframe: it takes a dataframe that concatenates the empty dataframe created with the stockprices
    # @stockprices_pivot_df: it creates a pivot table with tickers and adj.close stockprices in the stockprice dataframe
    # Return: It returns a dataframe with the stockprices populated
    ##
    stockprice_dataframe = pd.DataFrame()
    for ticker in tickers:
        stockprices = pdr.get_data_yahoo(ticker, start_date, end_date)
        stockprices ['ticker'] = ticker
        stockprice_dataframe = pd.concat([stockprice_dataframe, stockprices])
    stockprice_dataframe = stockprice_dataframe.reset_index()
    return stockprice_dataframe

stockprice_dataframe = get_stockprices(tickers)

#Question 5 - Merge the data queried data in 3 and 4
dataset = pd.read_json("https://www.sec.gov/files/company_tickers.json")
def get_tickers(dataset):
    ##
    # get_tickers - This is the get_tickers function
    #
    # Description: This function gets the tickers and company name of all the companies that file returns on SEC edgar database and adds it to a dataframe
    # @cik_df: it takes a dataframe that contains the cik, tickers and company name of all companies who file with the SEC
    # Return: It returns a dataframe with the cik and tickers populated
    ##
    datasetFrame = pd.DataFrame(dataset)
    cik_df = datasetFrame.T
    cik_df.rename(columns={'cik_str':'cik', 'title':'company'}, inplace=True)
    return cik_df

cik_df = get_tickers(dataset)

def merge_stockprice_filing(Queried_dataframe, cik_df):
    ##
    # merge_stockprice_filing - This is the merge_stockprice_filing function
    #
    # Description: This function gets the merges the 10-Q filings queried with the stockprice dataframe of the 41 firms selected
    # @cik_tickers_df: it takes a dataframe that merges the queried 10-Q filings with the cik_tickers dataframe
    # @company_stockprices_filings: This merges the stockprices with the queried 10-Q filings
    # Return: It returns a dataframe with the cik and tickers populated
    ##
    cik_tickers_df = pd.merge(Queried_dataframe, cik_df)
    cik_tickers_df.rename(columns={'date':'Date'}, inplace=True)
    cik_tickers_df['Date'] = pd.to_datetime(cik_tickers_df['Date'], format='%Y/%m/%d')
    return cik_tickers_df

cik_tickers_df = merge_stockprice_filing(Queried_dataframe, cik_df)
company_stockprices_filings = pd.merge(left=get_stockprices(tickers), right=merge_stockprice_filing(Queried_dataframe, cik_df), on=['Date', 'ticker'], how='left')


#Question 6 - Conduct textual analyses of the sampled 10-Q firms
##

company_stockprices_filings_with_tone = add_tone(company_stockprices_filings, Base_url)
# company_stockprices_filing_with_tone: This takes the add_tone function and adds filing tone to merged dataframe with the stockprices of the selected companies and their 10-Q filings

company_stockprices_filings_with_tone.to_csv("prices_full.csv")
# company_stockprices_filings_with_tone.rename(columns={'Adj Close':'Adj_Close'}, inplace=True)

company_stockprices_with_filings = company_stockprices_filings_with_tone.dropna()
# company_stockprices_with_filing: This filters the company_stockprices_filings_with_tone by dropping all the dates where no 10-Q filing was done.

#Question 7 - Summary statistics of the companies'stock characteristics

### Returns ###
def stockprice_returns(s_dataframe):
    ##
    # stockprice_returns- This is the stockprice_returns function
    #
    # Description: This function calculates the stockprice_returns and adds it to the dataframe
    # @s_dataframe: it takes a dataframe that needs returns added to it as parameter
    # Return: It returns a dataframe with the returns column populated
    ##
    ticker_list = s_dataframe['ticker'].tolist()
    adj_close_list = s_dataframe['Adj Close'].tolist()
    returns_list = []
    
    i = 0
    
    for tick in ticker_list:
        if not ((i-1)<0):
            if ticker_list[i-1] == ticker_list[i]:
                returns = np.log (adj_close_list[i-1] / adj_close_list[i])
                returns_list.append(-returns)
            else:
                returns_list.append(0)
        else:
            returns_list.append(0)
        i = i+1

    company_stockprices_filings_with_tone['returns'] = returns_list

stockprice_returns_df = stockprice_returns(company_stockprices_filings_with_tone)


##### Volatility ####
stockprice_pivot_df = company_stockprices_filings_with_tone.set_index('Date').pivot(columns='ticker', values='returns')
# This creates a pivoted dataframe of stockprice returns

describe_returns = stockprice_pivot_df.describe()
describe_returns.iloc[:, 36:41]
# This produces a summary statistics of the stockprice returns

describe = stockprice_pivot_df.iloc[:, :20]
describe.hist(bins = 100, label = 'Stockprice_Returns', alpha = 0.5, linewidth =1.5, figsize = (15,15))
plt.show()
# This plots a histogram of the stockprice returns. It also tests for the normality of the returns and tells us which stockprice is normally distributed or not normally distributed.


### Liquidity ###
company_stockprices_filings_with_tone['Market_Cap']= company_stockprices_filings_with_tone['Open'] * company_stockprices_filings_with_tone['Volume']
# This calculates the market capitalisation and inserts it inside the company_stockprices_filings_with_tone dataframe

Market_Cap_pivot_df = company_stockprices_filings_with_tone.set_index('Date').pivot(columns='ticker', values='Market_Cap')
# This creates a pivoted dataframe of only the date and the market capitalisation of the 41 stockprices

liquidity = Market_Cap_pivot_df.iloc[:, 30:41]
liquidity.plot(label = 'Market_Cap', figsize = (15,15))
plt.title('Market Capitalisation')
# This plots a graph of the market capitalisation and shows how the stockprices were traded when compared to other stocks.

Market_Cap_pivot_df.describe()
# This produces a summary statistics of the market capitalisations.


### Volume ###
Volume_pivot_df = company_stockprices_filings_with_tone.set_index('Date').pivot(columns='ticker', values='Volume')
# This creates a pivoted dataframe of only the date and the volume of the 41 stockprices

Volume_pivot_df.plot(label = 'Volume', figsize = (15,15))
plt.title('Volume Traded')
# This plots a graph of the volume of stocks traded and compares it to that of other stocks.

Volume_pivot_df.describe()
# This produces a summary statistics of the volume traded.


#Question 8 - Econometric Procedure examining the relationship of the stock price characteristics to its tone in 10-Q filings

yf.pdr_override()
tickers = ['^GSPC', '^IRX']
start_date = '2021-01-01'
end_date = '2021-12-31'

data = get_stockprices(tickers)
# This calls the get_stockprices function above and download the stockprices for S&P 500 index and Tbills yield Index.

def log_change(price):
    ##
    # log_change - This is the logchange function
    #
    # Description: This function gets calculates the log returns of the market return and the riskfree return
    # @data_1: it takes a pivot of the adjusted close figures and uses it to calculate the returns
    # Return: It returns  columns in the dataframe that can be used for the regression
    ##
    return np.log(price/price.shift(1))
data1 = data.set_index('Date').pivot(columns = 'ticker', values = 'Adj Close')
data1['Rm'] = 100*log_change(data1['^GSPC'])
data1['Rf'] = data1['^IRX']/250
data1['Rm_Rf'] = data1['Rm'] - data1['Rf']
Rm_data = data1.drop(data1.columns[[0, 1]], axis=1, inplace=True)


def clean_reg_data(r_dataframe):
    ##
    # clean_reg_data - This is the clean_reg_data function
    #
    # Description: This function gets selected columns from the main dataframe and merges it with the data1 dataframe
    # @Reg_df: it takes selected columns in the dataframe and converts the returns column to a percentage
    # Return: It returns a a merged dataframe with that has been cleaned and can be used for the regression
    ##
    Reg_df = r_dataframe.set_index('Date')
    Reg_df = Reg_df[['ticker', 'returns', 'tone']]   
    Reg_df['returns']= 100*Reg_df['returns']
    Regression_df = pd.merge(left=Reg_df, right=data1, on=['Date'], how='left')
    
    return Regression_df

Regression_df = clean_reg_data(company_stockprices_filings_with_tone)

Regression_df ['tone'] = Regression_df ['tone'].replace(np.nan, 0)
Regression_df ['tone'] = Regression_df ['tone'].replace('inaccessible', 0)

def clean_tone(t_dataframe):
    ##
    # clean_tone - This is the clean_tone function
    #
    # Description: This function assigns the dummy variable 1 for all the negative filing tones
    # Return: It returns a dataframe with a cleaned tone. 0 for days with no filing and 1 for filing days
    ##
    
    tone_list = t_dataframe['tone'].tolist()   
    i = 0

    for i in range(0,len(tone_list)):
        if (tone_list[i] < 0):
            tone_list[i] = 1              
        i = i + 1
    
    Regression_df ['tone'] = tone_list    
             
Regression_df2 = clean_tone(Regression_df)


Regression_df3 = Regression_df.reset_index(level=0)
Regression_df3 = Regression_df3.iloc[251:10291, :7]

def regression_function(reg_dataframe):
    ##
    # regression_function - This is the regression function
    #
    # Description: This function calculates the regression using the CAPM factors and the tone
    # formula: This is the regression formular that specifies the dependent and independent variables
    # Return: It returns the regression result
    ##
    formula = 'returns ~ Rm_Rf + Rf + tone'
    results = smf.ols(formula, reg_dataframe).fit()
    
    return results
    
Regression = regression_function(Regression_df3)
print(Regression.summary())
