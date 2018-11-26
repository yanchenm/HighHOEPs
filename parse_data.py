import os
import re

import xml.etree.ElementTree as ET

import pandas as pd


def get_real_mkt_price(root_dir):
    root = '{}/csv/RealtimeMktPrice/'.format(root_dir)
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

    data.to_csv('{}/output/RealtimeMktPrice.csv'.format(root_dir), index=False)


def get_real_mkt_totals(root_dir):
    root = '{}/csv/RealtimeMktTotals/'.format(root_dir)
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

    data.to_csv('{}/output/RealtimeMktTotals.csv'.format(root_dir), index=False)


def get_predisp_mkt_price(root_dir):
    root = '{}/csv/PredispMktPrice/'.format(root_dir)
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

    data.to_csv('{}/output/PredispMktPrice.csv'.format(root_dir), index=False)


def get_predisp_mkt_totals(root_dir):
    root = '{}/csv/PredispMktTotals/'.format(root_dir)
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

    data.to_csv('{}/output/PredispMktTotals.csv'.format(root_dir), index=False)


def get_hoep(root_dir):
    root = '{}/csv/DispUnconsHOEP/'.format(root_dir)
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

    data.to_csv('{}/output/HOEP.csv'.format(root_dir), index=False)


def get_forecasts(root_dir):
    columns = ['update_time','MP_type','fuel_type','zone','date','hour','MW_forecast']

    data = pd.DataFrame(columns=columns)

    i = 0

    for filename in os.listdir('{}/xml/VGForecastSummary'.format(root_dir)):
        i += 1
        if "_v" not in filename: continue
        print(filename)
        tree = ET.parse('{}/xml/VGForecastSummary/'.format(root_dir)+filename)
        root = tree.getroot()

        #if i > 100: break

        for timestamp in root[1]:
            #print(timestamp.tag,timestamp.text)
            if timestamp.text != '\n':
                update_time = timestamp.text
            for MP_type in timestamp:
                #print("\t",MP_type.tag,MP_type.text)
                if MP_type.text != '\n':
                    MP = MP_type.text
                    if MP == 'EMBEDDED':break
                for fuel_type in MP_type:
                    #print("\t\t", fuel_type.tag,fuel_type.text)
                    if fuel_type.text != '\n':
                        fuel = fuel_type.text
                    for zone in fuel_type:
                        #print("\t\t\t", zone.tag,zone.text)
                        if zone.text != '\n':
                            area = zone.text
                            if area != "OntarioTotal": break
                            j = 0
                        for date in zone:
                            #print("\t\t\t\t", date.tag,date.text)
                            if j>5:break
                            if date.text != '\n':
                                forecast_date = date.text
                            else:
                                forecast_hour = date[0].text
                                forecast_MW = date[1].text
                                d = dict(zip(columns,[update_time,MP,fuel,area,forecast_date,forecast_hour,forecast_MW]))
                                data = data.append(d,ignore_index=True)
                                j+=1

    data['hour'] = data['hour'].astype(int)-1
    data['DateTime'] = data['date']+ ' ' + data['hour'].astype(str) + ':00'
    data['update_time'] = pd.to_datetime(data['update_time'])
    data['update_time'] = data['update_time'].dt.ceil('h')
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['PD_hours_back'] = (data['DateTime'] - data['update_time']).dt.components.hours
    data.to_csv('{}/output/VGForecasts.csv'.format(root_dir))


def get_gen_output(root_dir):
    i = 10
    columns = 'date,hour,fuel,generator,field,value'.split(',')
    data = pd.DataFrame(columns=columns)

    for filename in os.listdir('{}/xml/GenOutputCapability'.format(root_dir)):

        if "_v" in filename: continue
        if filename == 'PUB_GenOutputCapability_20181017.xml': i = 0
        # if i > 5: continue                                          #GET RID OF THIS LINE to do all files
        i += 1
        print(filename)
        tree = ET.parse('{}/xml/GenOutputCapability/'.format(root_dir) + filename)
        root = tree.getroot()

        for date in root[1]:
            #print(date.tag,date.text)
            if 'Date' in date.tag:
                date_value = date.text
            for generator in date:
                #print('\t',generator.tag, generator.text)
                for name_fuel_output_capability_capacity in generator:
                    #print('\t\t',name_fuel_output_capability_capacity.tag, name_fuel_output_capability_capacity.text)
                    if 'Name' in name_fuel_output_capability_capacity.tag:
                        gen_name = name_fuel_output_capability_capacity.text
                    if 'Fuel' in name_fuel_output_capability_capacity.tag:
                        fuel = name_fuel_output_capability_capacity.text
                    for field in name_fuel_output_capability_capacity:
                        #print('\t\t\t', field.tag, field.text)
                        field_val = field.tag.split('}')[-1]
                        if field_val != 'Output':continue
                        if len(field) == 1: continue
                        hour = field[0].text
                        MW = field[1].text
                        #print('\t\t\t\t hour:{}'.format(hour),'MW:{}'.format(MW))
                        data = data.append(dict(zip(columns,[date_value,hour,fuel,gen_name,field_val,MW])),ignore_index=True)

    data['hour'] = data['hour'].astype(int) - 1
    data['DateTime'] = data['date'] + ' ' + data['hour'].astype(str) + ':00'
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.to_csv('{}/output/GenOutputCapability.csv'.format(root_dir))


if __name__ == '__main__':
    get_real_mkt_price('./data/test')
    get_real_mkt_totals('./data/test')
    get_predisp_mkt_price('./data/test')
    get_predisp_mkt_totals('./data/test')
    get_hoep('./data/test')
    get_gen_output('./data/test')
    get_forecasts('./data/test')
