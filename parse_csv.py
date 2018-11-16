import os
import re

import pandas as pd


def get_real_mkt_price():
    root = './data/csv/RealtimeMktPrice/'
    csv_columns = ['hour', 'interval', 'type', 'zone', 'price', 'code']

    data = pd.DataFrame(columns=['datetime', 'interval', 'type', 'zone', 'price', 'code'])

    file_pattern = r'PUB_RealtimeMktPrice_([0-9]{8})[0-9]{2}.csv'

    for filename in os.listdir(root):
        if '_v' in filename: continue

        temp = pd.read_csv(root + filename, header=None, names=csv_columns, skiprows=4)

        print(filename)

        date = re.search(file_pattern, filename).group(1)

        temp['hour'] = temp['hour'] - 1
        temp['hour'] = date + temp['hour'].astype(str)
        temp['datetime'] = pd.to_datetime(temp['hour'], format='%Y%m%d%H')

        temp = temp.drop(columns='hour')

        # Move datetime column first
        temp = temp[['datetime', 'interval', 'type', 'zone', 'price', 'code']]

        data = data.append(temp, ignore_index=True)

    data.to_csv('./data/output/RealtimeMktPrice.csv', index=False)


def get_real_mkt_totals():
    root = './data/csv/RealtimeMktTotals/'
    csv_columns = ['hour', 'interval', 'total_energy', 'total_10s', 'total_10n', 'total_30r', 'total_loss',
                   'total_load', 'total_disp_load', 'code']

    data = pd.DataFrame(columns=['datetime', 'interval', 'total_energy', 'total_10s', 'total_10n', 'total_30r',
                                 'total_loss', 'total_load', 'total_disp_load', 'code'])

    file_pattern = r'PUB_RealtimeMktTotals_([0-9]{8})([0-9]{2}).csv'

    for filename in os.listdir(root):
        if '_v' in filename: continue

        temp = pd.read_csv(root + filename, header=None, names=csv_columns, skiprows=4)

        print(filename)

        date = re.search(file_pattern, filename).group(1)

        temp['hour'] = temp['hour'] - 1
        temp['hour'] = date + temp['hour'].astype(str)
        temp['datetime'] = pd.to_datetime(temp['hour'], format='%Y%m%d%H')

        temp = temp.drop(columns='hour')

        # Move datetime column first
        temp = temp[['datetime', 'interval', 'total_energy', 'total_10s', 'total_10n', 'total_30r',
                     'total_loss', 'total_load', 'total_disp_load', 'code']]

        data = data.append(temp, ignore_index=True)

    data.to_csv('./data/output/RealtimeMktTotals.csv', index=False)


def get_predisp_mkt_price():
    root = './data/csv/PredispMktPrice/'
    csv_columns = ['hour', 'zero', 'type', 'zone', 'price']

    data = pd.DataFrame(columns=['update_datetime', 'datetime', 'type', 'zone', 'price', 'PD_hours_back'])

    pattern = r'\\\\CREATED AT (.{10}) ([0-9]{2}).{6} FOR (.{10})'

    for filename in os.listdir(root):
        if '_v' not in filename: continue

        # Find creation and prediction dates and times in file
        file = open(root + filename, 'r')

        for line in file:
            for match in re.finditer(pattern, line):
                if match is not None:
                    update_date = match.group(1)
                    update_time = match.group(2)
                    date = match.group(3)

                    update_datetime = update_date + update_time

        file.close()

        # Remove terminating semicolons
        with open(root + filename, 'r') as f:
            new_text = f.read()

            while ';' in new_text:
                new_text = new_text.replace(';', '')

        with open(root + filename, 'w') as f:
            f.write(new_text)

        temp = pd.read_csv(root + filename, header=None, names=csv_columns, skiprows=4)

        print(filename)

        temp['hour'] = temp['hour'] - 1
        temp['hour'] = date + temp['hour'].astype(str)
        temp['datetime'] = pd.to_datetime(temp['hour'], format='%Y/%m/%d%H')
        temp['update_datetime'] = update_datetime
        temp['update_datetime'] = pd.to_datetime(temp['update_datetime'], format='%Y/%m/%d%H')
        temp['PD_hours_back'] = (temp['datetime'] - temp['update_datetime'])
        temp['PD_hours_back'] = temp['PD_hours_back'].dt.components.days * 24 + temp['PD_hours_back'].dt.components.hours

        # Move datetime column first
        temp = temp[['update_datetime', 'datetime', 'type', 'zone', 'price', 'PD_hours_back']]

        data = data.append(temp, ignore_index=True)

    data.to_csv('./data/output/PredispMktPrice.csv', index=False)


def get_predisp_mkt_totals():
    root = './data/csv/PredispMktTotals/'
    csv_columns = ['hour', 'zero', 'total_energy', 'total_10s', 'total_10n', 'total_30r', 'total_loss',
                   'total_load', 'total_disp_load']

    data = pd.DataFrame(columns=['update_datetime', 'datetime', 'total_energy', 'total_10s', 'total_10n', 'total_30r',
                                 'total_loss', 'total_load', 'total_disp_load', 'PD_hours_back'])

    pattern = r'\\\\CREATED AT (.{10}) ([0-9]{2}).{6} FOR (.{10})'

    for filename in os.listdir(root):
        if '_v' not in filename: continue

        # Find creation and prediction dates and times in file
        file = open(root + filename, 'r')

        for line in file:
            for match in re.finditer(pattern, line):
                if match is not None:
                    update_date = match.group(1)
                    update_time = match.group(2)
                    date = match.group(3)

                    update_datetime = update_date + update_time

                    break

        file.close()

        temp = pd.read_csv(root + filename, header=None, names=csv_columns, skiprows=4)

        print(filename)

        temp['hour'] = temp['hour'] - 1
        temp['hour'] = date + temp['hour'].astype(str)
        temp['datetime'] = pd.to_datetime(temp['hour'], format='%Y/%m/%d%H')
        temp['update_datetime'] = update_datetime
        temp['update_datetime'] = pd.to_datetime(temp['update_datetime'], format='%Y/%m/%d%H')
        temp['PD_hours_back'] = (temp['datetime'] - temp['update_datetime'])
        temp['PD_hours_back'] = temp['PD_hours_back'].dt.components.days * 24 + temp['PD_hours_back'].dt.components.hours

        # Move datetime column first
        temp = temp[['update_datetime', 'datetime', 'total_energy', 'total_10s', 'total_10n', 'total_30r',
                     'total_loss', 'total_load', 'total_disp_load', 'PD_hours_back']]

        data = data.append(temp, ignore_index=True)

    data.to_csv('./data/output/PredispMktTotals.csv', index=False)


def get_hoep():
    root = './data/csv/DispUnconsHOEP/'
    csv_columns = ['hour', 'hoep', 'source']

    data = pd.DataFrame(columns=['datetime', 'hoep'])

    file_pattern = r'PUB_DispUnconsHOEP_([0-9]{8}).csv'

    for filename in os.listdir(root):
        if '_v' in filename: continue

        temp = pd.read_csv(root + filename, header=None, names=csv_columns, skiprows=4)

        print(filename)

        date = re.search(file_pattern, filename).group(1)

        temp['hour'] = temp['hour'] - 1
        temp['hour'] = date + temp['hour'].astype(str)
        temp['datetime'] = pd.to_datetime(temp['hour'], format='%Y%m%d%H')

        temp = temp.drop(columns=['hour', 'source'])

        # Move datetime column first
        temp = temp[['datetime', 'hoep']]

        data = data.append(temp, ignore_index=True)

    data.to_csv('./data/output/HOEP.csv', index=False)


if __name__ == '__main__':
    # get_real_mkt_price()
    # get_real_mkt_totals()
    get_predisp_mkt_price()
    # get_predisp_mkt_totals()
    # get_hoep()
