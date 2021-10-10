import streamlit as st
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pyecharts
import random
import requests
import streamlit as st
from pandas.api.types import CategoricalDtype
from pyecharts.charts import Bar
from pyecharts.faker import Faker
import streamlit_echarts
from pyecharts import options as opts
from pyecharts.globals import ThemeType


@st.cache
def get_weekday(dt):
    return dt.weekday()


def get_hour(dt):
    return dt.hour


def split_datetime(dt):
    return str(dt.split()[:2][0]) + '-2020'


def split_date(dt):
    if dt.split()[2] == 'on':
        return dt.split()[3]
    if dt.split()[2] == '??Monday':
        return 'Monday'
    return dt.split()[2]


def split_month(dt):
    return dt.split()[0].split('-')[0]


def split_jour(dt):
    j = int(dt.split()[1].split(':')[0])
    if j <= 6:
        return 'midnight'
    if j <= 11:
        return 'morning'
    if j <= 14:
        return 'noon'
    if j <= 18:
        return 'afternoon'
    if j <= 24:
        return 'evening'


def count_rows(rows):
    return len(rows)


def load_data():
    path = r"data/data(tranlated_utf-8).csv"
    df = pd.read_csv(path, delimiter=',')
    df = df

    path = r"converted_data.csv"
    df1 = pd.read_csv(path, delimiter=',')
    df1.head(1)
    return df, df1


def split_t(dt):
    if dt.split()[2] == 'on':
        return dt.split()[3]
    if dt.split()[2] == '??Monday':
        return 'Monday'
    return dt.split()[2]


def df1_processing(df1, df):
    start_1 = df1['cmplt_start_address_converted'].apply(lambda x: float(x.split(',')[0])).to_list()
    start_2 = df1['cmplt_start_address_converted'].apply(lambda x: float(x.split(',')[1])).to_list()
    end_1 = df1['cmplt_ending_address_converted'].apply(lambda x: float(x.split(',')[0])).to_list()
    end_2 = df1['cmplt_ending_address_converted'].apply(lambda x: float(x.split(',')[1])).to_list()
    date = df['DATE'].to_list()
    df1_temp = pd.DataFrame(
        {'latitude': end_2, 'longitude': end_1, 'p': df['end'].to_list(),
         't': df['Time to get on the train'].to_list()})
    df1_temp = df1_temp.sort_values(by=['latitude'])
    df_temp = pd.DataFrame({'latitude': start_2, 'longitude': start_1, 'p': df['starting point'].to_list(),
                            't': df['Time to get on the train'].to_list()})
    df_temp = df_temp.sort_values(by=['latitude'])
    df1_px = df1_temp
    df1_px['week'] = df1_px['t'].map(split_t)
    df_px = df_temp
    df_px['week'] = df_px['t'].map(split_t)
    df_px['p_or_d'] = 'pick up'
    df1_px['p_or_d'] = 'drop off'
    result = pd.concat([df_px, df1_px])
    return result


def app():
    items_list = [
        'Dataset',
        'The Code of Webdriver'
    ]
    items_type = st.sidebar.selectbox(
        " Which kind would you like  to explore?",
        items_list
    )
    if items_type == 'Dataset':
        st.title("Dataset")
        path = r"data/data(tranlated_utf-8).csv"
        df = pd.read_csv(path, delimiter=',')
        df['datetime'] = df['Time to get on the train'].map(split_datetime)
        df['datetime'] = df['datetime'].map(pd.to_datetime)

        df_g_temp = df.drop(['Serial number'], axis=1)
        st.title("Skim what the data looks like:")
        st.write(df_g_temp)
        st.title("Check for missing data")
        st.pyplot(msno.bar(df).figure)
        st.pyplot(msno.matrix(df).figure)

    if items_type == 'The Code of Webdriver':
        st.title('Convert text address to latitude and longitude with Chrome Driver')
        with st.echo():
            from bs4 import BeautifulSoup
            import time
            from selenium import webdriver
            from selenium.webdriver.common.keys import Keys
            import re
            import win32clipboard as w
            import win32con

            def getText():  # Read clipboard
                w.OpenClipboard()
                d = w.GetClipboardData(win32con.CF_TEXT)
                w.CloseClipboard()
                return d

            def Convert(ad):
                ls = []
                chrome_driver = 'D:\chromedriver\92\chromedriver.exe'  # the address of chrome_driver
                browser = webdriver.Chrome(executable_path=chrome_driver)
                browser.get(r'https://lbs.amap.com/tools/picker')  # requests
                for i in ad:
                    elem_user = browser.find_element_by_name("search")
                    browser.find_element_by_name("search").send_keys(Keys.CONTROL, "a")
                    browser.find_element_by_name("search").send_keys(Keys.BACKSPACE)
                    elem_user.send_keys(i)
                    browser.find_element_by_class_name('picker-btn.btn-search').click()
                    browser.find_element_by_class_name('picker-btn.picker-copy').click()
                    ss = getText().decode('GB2312')
                    ls.append(ss)
                    time.sleep(0.1)
                browser.close()
                return pd.DataFrame(ls)

            # r"data/data(zh_utf-8).csv"

            path = r"./data/demo-coor.csv"
            dff = pd.read_csv(path, delimiter=',')
            dff.head()
            # dff['cmplt_start_address_converted'] = Convert(dff['cmplt_start_address'].to_list())

        if st.button('Run'):
            st.write(dff['cmplt_start_address'].to_list())
            dff['cmplt_start_address_converted'] = Convert(dff['cmplt_start_address'].to_list())
            st.write(dff['cmplt_start_address_converted'].to_list())
            #dff = Convert(dff['cmplt_start_address'].to_list())
            st.write(dff)
            st.button('Clean')
