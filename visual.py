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
import plotly.figure_factory as ff
import plotly.express as px
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType


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



def rename(ad):
    # Since the user will define a 'current location' for easier input, and mine is this address, I need to convert it.
    if ad == 'current position':
        ad = 'Guanyin Temple - North Gate'
    return ad


def app():
    st.title("Visualization")
    st.subheader("The datasets:")
    df, df1 = load_data()
    df['datetime'] = df['Time to get on the train'].map(split_datetime)
    df['datetime'] = df['datetime'].map(pd.to_datetime)
    df['DATE'] = df['Time to get on the train'].map(split_date)

    df1 = df1_processing(df1, df)

    # create a CategoricalType
    cat_DATE_order = CategoricalDtype(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True
    )
    df['DATE'] = df['DATE'].astype(cat_DATE_order)  # Forced conversion type
    data_date = df.groupby(['DATE']).apply(count_rows)  # Get the number of each date
    # data_date.to_frame().sort_values('DATE', inplace=True)
    # data_date


    event_list = df1["week"].unique()
    event_list = np.insert(event_list, 0, ["ALL"], axis=0)
    event_type = st.sidebar.selectbox(
        "Which weekday do you want to explore?",
        event_list
    )

    points_list = df1["p_or_d"].unique()
    points_list = np.insert(points_list, 0, ["ALL"], axis=0)

    points_type = st.sidebar.selectbox(
        " Do you want to explore pick-up points or drop-off points?",
        points_list
    )

    if event_type == 'ALL' and points_type == 'ALL':
        part_df1 = df1
        st.write(df.head(3))
        st.write(f"According to your filter, the data contains  {len(part_df1)} rows")
        st.title(f"Overview")
        st.map(part_df1)
    elif event_type != 'ALL' and points_type == 'ALL':
        part_df1 = df1[(df1["week"] == event_type)]
        st.write(df[(df["DATE"] == event_type)].head(3))
        st.write(f"According to your filter, the data contains  {len(part_df1)} rows")
        st.title(f"{part_df1.iloc[1].values.tolist()[-2]}- pick-up & drop-off ")
        st.map(part_df1)
    elif event_type == 'ALL' and points_type != 'ALL':
        part_df1 = df1[(df1["p_or_d"] == points_type)]
        st.write(df[(df["DATE"] == event_type)].head(3))
        st.write(f"According to your filter, the data contains  {len(part_df1)} rows")
        st.title(f"ALL weekday- {part_df1.iloc[1].values.tolist()[-1]} points")
        st.map(part_df1)
    else:
        part_df1 = df1[(df1["week"] == event_type) & (df1["p_or_d"] == points_type)]
        st.write(df[(df["DATE"] == event_type)].head(3))
        st.write(f"According to your filter, the data contains  {len(part_df1)} rows")
        st.title(f"{part_df1.iloc[1].values.tolist()[-2]}- {part_df1.iloc[1].values.tolist()[-1]} points")
        st.map(part_df1)

    # Cause I want to Sort by week,BUT they are string, To deal it,I create a Sorting rule

    # create a CategoricalDtype
    cat_DATE_order = CategoricalDtype(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True
    )
    df['DATE'] = df['DATE'].astype(cat_DATE_order)  # Forced conversion type
    data_date = df.groupby(['DATE']).apply(count_rows)  # Get the number of each date
    # data_date.to_frame().sort_values('DATE', inplace=True)
    # Take Time_Period[morning/afternoon etc] out of ‘Time to get on the train’
    df['Time_Period'] = df['Time to get on the train'].map(split_jour)
    #############################################################################
    # create a CategoricalDtype
    cat_tp_order = CategoricalDtype(
        ['morning', 'noon', 'afternoon', 'evening', 'midnight'],
        ordered=True
    )
    df['Time_Period'] = df['Time_Period'].astype(cat_tp_order)
    data_tp = df.groupby(['Time_Period']).apply(count_rows)
    # data_tp.to_frame().sort_values('Time_Period', inplace=True)
    df['Time_Period'] = df['Time_Period'].astype(cat_tp_order)
    data_tp = df.groupby(['Time_Period']).apply(count_rows)

    ###################################################################
    df['MONTH'] = df['Time to get on the train'].map(split_month)  # Take MONTH out of ‘Time to get on the train’
    # Since there is no Tuesday in the dataset, I need to add an element for Tuesday
    data_month = df.groupby(['MONTH']).apply(count_rows)  # group by month
    # Creating the block Series
    m2 = pd.Series([0])
    # Create the block Index
    index_2 = ['02']
    # set the index of block series
    m2.index = index_2
    data_month = data_month.append(m2)
    data_month = data_month.sort_values(key=lambda x: x.index)
    tp = df.groupby('Time_Period').sum()
    ###########################################################################
    df['starting point'] = df['starting point'].map(rename)
    df['end'] = df['end'].map(rename)
    # Using the dictionary, get the number of times each address
    Dic = {}
    for i in df['starting point']:
        Dic[i] = Dic.get(i, 0) + 1

    startls = list(Dic.items())
    startls.sort(key=lambda item: item[1], reverse=True)

    Dic = {}
    for i in df['end']:
        Dic[i] = Dic.get(i, 0) + 1

    endls = list(Dic.items())
    endls.sort(key=lambda item: item[1], reverse=True)
    ##########################################################################

    ###########################################################################
    ###########################################################################
    ###########################################################################
    items_list = [
        'Scatter-coordinates',
        'Frequency by DoM-DIDI-2020',
        'Frequency by MONTH-DIDI-2020',
        'Frequency by Time_Period-DIDI-2020',
        'Spending & Mileage of Traveling-DIDI',
        'Wordcloud Drop-off',
        'Wordcloud Pick-up'
    ]
    items_type = st.sidebar.selectbox(
        " Which kind of visual do you want to explore?",
        items_list
    )
    ###########################################################################
    if items_type == 'Frequency by Time_Period-DIDI-2020':
        st.title('Frequency by Time_Period-DIDI-2020-BAR')
        fig = plt.figure()
        plt.bar(range(1, 6), data_tp)
        plt.xticks(range(1, 6), data_tp.index)
        plt.xlabel('Time_Period')
        plt.ylabel('Frequency')
        plt.title('Frequency by Time_Period -DIDI - 2020')
        st.plotly_chart(fig)

    if items_type == 'Frequency by DoM-DIDI-2020':
        st.title('Frequency by DoM-DIDI-2020-BAR')
        fig1 = plt.figure()
        plt.bar(range(1, 8), data_date)
        plt.xticks(range(1, 8), data_date.index)
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.title('Frequency by DoM -DIDI - 2020')
        st.plotly_chart(fig1)

    if items_type == 'Frequency by MONTH-DIDI-2020':
        st.title("Frequency by MONTH-DIDI-2020-BAR")

        fig = plt.figure()
        plt.bar(range(1, 13), data_month)
        plt.xticks(range(1, 13), data_month.index)
        plt.xlabel('Month')
        plt.ylabel('Frequency')
        plt.title('Frequency by MONTH -DIDI - 2020')
        st.plotly_chart(fig)

    if items_type == 'Spending & Mileage of Traveling-DIDI':
        from pyecharts import options as opts
        st.title("Spending & Mileage of Traveling-DIDI")

        c = (
            Bar()
                .add_xaxis(tp.index.to_list())  # [LIST],Time_Period
                .add_yaxis("The amount [CNY]", tp['The amount [yuan]'].to_list(), stack="stack1")
                .add_yaxis("Mileage[km]", tp['Mileage [km]'].to_list(), stack="stack1")
                .set_series_opts(label_opts = opts.LabelOpts(is_show=False))
            # .set_global_opts(title_opts=opts.TitleOpts(title="Spending & Mileage of Traveling"))
            # if wanna render on a new page use the following
            # .render("Spending & Mileage of Traveling by DoM -DIDI -2020.html")
        )
        streamlit_echarts.st_pyecharts(c)

    if items_type == 'Wordcloud Drop-off':
        st.title("Wordcloud Drop-off")
        from pyecharts.charts import WordCloud
        c = (
            WordCloud()
                .add("", endls, shape=SymbolType.DIAMOND)
                #.set_global_opts(title_opts=opts.TitleOpts(title="drop off"))
            # .render("wordcloud_end.html")
        )
        c
        streamlit_echarts.st_pyecharts(c)

    if items_type == 'Wordcloud Pick-up':
        from pyecharts import options as opts
        from pyecharts.charts import WordCloud
        st.title("Wordcloud Pick-up")
        c = (
            WordCloud()
                .add(
                "",
                startls,

                textstyle_opts=opts.TextStyleOpts(font_family="cursive"),
            )
                .set_global_opts(title_opts=opts.TitleOpts(title="pick up"))
            # .render("wordcloud_start.html")
        )
        streamlit_echarts.st_pyecharts(c)


    if items_type == 'Scatter-coordinates':
        st.subheader('Scatter-coordinates')
        # fig = plt.figure()
        fig = px.scatter(part_df1, x="latitude", y="longitude", log_x=True, color='p_or_d',
                         hover_name="p", hover_data=["latitude", "longitude", "t"])

        st.plotly_chart(fig)
