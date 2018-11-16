import os

import pandas as pd
from datetime import timedelta, datetime

from sklearn.preprocessing import StandardScaler


def manipulate():

    if os.path.isfile('./data/output/final_data.csv'):
        return

    curr_datetime = datetime(year=2018, month=10, day=13, hour=0)
    end_datetime = datetime(year=2018, month=11, day=13, hour=0)

    # Data points where not all fields are populated (to be skipped)
    broken_data = [datetime(year=2018, month=10, day=23, hour=15),
                   datetime(year=2018, month=10, day=24, hour=9),
                   datetime(year=2018, month=10, day=24, hour=11),
                   datetime(year=2018, month=10, day=25, hour=9),
                   datetime(year=2018, month=10, day=25, hour=18),
                   datetime(year=2018, month=10, day=29, hour=11),
                   datetime(year=2018, month=10, day=29, hour=21),
                   datetime(year=2018, month=11, day=12, hour=23)]

    hour = timedelta(hours=1)

    # Load in processed csv files
    gen_output = pd.read_csv('./data/output/GenOutputCapability.csv')
    forecasts = pd.read_csv('./data/output/VGForecasts.csv')
    real_mkt_price = pd.read_csv('./data/output/RealtimeMktPrice.csv')
    real_mkt_totals = pd.read_csv('./data/output/RealtimeMktTotals.csv')
    predisp_mkt_price = pd.read_csv('./data/output/PredispMktPrice.csv')
    predisp_mk_totals = pd.read_csv('./data/output/PredispMktTotals.csv')
    hoep = pd.read_csv('./data/output/HOEP.csv')

    # Final data fields
    columns = ['timestamp', 'hoep', 'mcp_1', 'mcp_2', 'mcp_3', 'mcp_4', 'mcp_5', 'mcp_6', 'mcp_7', 'mcp_8', 'mcp_9',
               'mcp_10', 'mcp_11', 'mcp_12', 'price_pd_1', 'price_pd_2', 'price_pd_3', 'price_pd_4', 'price_pd_5',
               'rt_market_totals_1', 'rt_market_totals_2', 'rt_market_totals_3', 'rt_market_totals_4',
               'rt_market_totals_5', 'rt_market_totals_6', 'rt_market_totals_7', 'rt_market_totals_8',
               'rt_market_totals_9', 'rt_market_totals_10', 'rt_market_totals_11', 'rt_market_totals_12',
               'market_totals_pd1', 'market_totals_pd2', 'market_totals_pd3', 'market_totals_pd4', 'market_totals_pd5',
               'wind_output', 'wind_pd1', 'wind_pd2', 'wind_pd3', 'wind_pd4', 'wind_pd5',
               'solar_output', 'solar_pd1', 'solar_pd2', 'solar_pd3', 'solar_pd4', 'solar_pd5',
               'gas_output', 'hydro_output', 'nuclear_output']

    data = pd.DataFrame(columns=columns)

    while not curr_datetime == end_datetime:

        # Skip manipulation for data known to be missing
        missing_start = datetime(year=2018, month=10, day=30, hour=12)
        missing_end = datetime(year=2018, month=11, day=1, hour=12)

        if (curr_datetime in broken_data) or (missing_start <= curr_datetime <= missing_end):
            curr_datetime += hour
            continue

        data_row = dict.fromkeys(columns)
        time_str = datetime.strftime(curr_datetime, '%Y-%m-%d %H:%M:%S')

        print(time_str)

        # Timestamp
        data_row['timestamp'] = time_str

        # HOEP
        data_row['hoep'] = float((hoep.loc[hoep.datetime == time_str])['hoep'])

        # Real-time market price
        for i in range(1, 13):
            col_name = 'mcp_{}'.format(i)
            data_row[col_name] = float((real_mkt_price.loc[real_mkt_price.datetime == time_str].query(
                'interval == {} and type == "ENGY" and zone == "ONZN"'.format(i)))['price'])

        # Predispatch price
        for i in range(1, 6):
            col_name = 'price_pd_{}'.format(i)
            data_row[col_name] = float((predisp_mkt_price.loc[predisp_mkt_price.datetime == time_str].query(
                'PD_hours_back == {} and type == "ENGY" and zone == "ONZN"'.format(i)))['price'])

        # Real-time market totals
        for i in range(1, 13):
            col_name = 'rt_market_totals_{}'.format(i)
            data_row[col_name] = float((real_mkt_totals.loc[real_mkt_totals.datetime == time_str].query(
                'interval == {}'.format(i)))['total_energy'])

        # Predispatch market totals
        for i in range(1, 6):
            col_name = 'market_totals_pd{}'.format(i)
            data_row[col_name] = float((predisp_mk_totals.loc[predisp_mk_totals.datetime == time_str].query(
                'PD_hours_back == {}'.format(i)))['total_energy'])

        # Wind
        data_row['wind_output'] = sum((gen_output.loc[gen_output.DateTime == time_str].query(
            'field == "Output" and fuel == "WIND"'))['value'])

        for i in range(1, 6):
            col_name = 'wind_pd{}'.format(i)
            data_row[col_name] = float((forecasts.loc[forecasts.DateTime == time_str].query(
                'zone == "OntarioTotal" and PD_hours_back == {} and fuel_type == "Wind"'.format(i)))['MW_forecast'])

        # Solar
        data_row['solar_output'] = sum((gen_output.loc[gen_output.DateTime == time_str].query(
            'field == "Output" and fuel == "SOLAR"'))['value'])

        for i in range(1, 6):
            col_name = 'solar_pd{}'.format(i)
            data_row[col_name] = float((forecasts.loc[forecasts.DateTime == time_str].query(
                'zone == "OntarioTotal" and PD_hours_back == {} and fuel_type == "Solar"'.format(i)))['MW_forecast'])

        # Gas
        data_row['gas_output'] = sum((gen_output.loc[gen_output.DateTime == time_str].query(
            'field == "Output" and fuel == "GAS"'))['value'])

        # Nuclear
        data_row['nuclear_output'] = sum((gen_output.loc[gen_output.DateTime == time_str].query(
            'field == "Output" and fuel == "NUCLEAR"'))['value'])

        # Hydro
        data_row['hydro_output'] = sum((gen_output.loc[gen_output.DateTime == time_str].query(
            'field == "Output" and fuel == "HYDRO"'))['value'])

        data = data.append(data_row, ignore_index=True)
        curr_datetime += hour

    data.to_csv('./data/output/final_data.csv', index=False)


def normalize():
    if os.path.isfile('./data/output/normalized_data.csv'):
        return

    data = pd.read_csv('./data/output/final_data.csv')

    # Separate timestamps from numeric data
    labels = data.timestamp
    data = data[data.columns[1:51]]

    # Apply z-score standardization to normalize data
    x = data.values
    transformer = StandardScaler()
    normalized_x = transformer.fit_transform(x)

    # Recombine data back into original format
    normalized_data = pd.DataFrame(normalized_x, columns=data.columns)
    normalized_data['timestamp'] = labels

    # Move timestamp field to start of dataframe
    columns = normalized_data.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    normalized_data = normalized_data[columns]

    normalized_data.to_csv('./data/output/normalized_data.csv', index=False)


if __name__ == '__main__':
    manipulate()
    normalize()
