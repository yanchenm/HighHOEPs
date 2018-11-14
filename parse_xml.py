import xml.etree.ElementTree as ET
import pandas as pd
import os

def get_forecasts():
    columns = ['update_time','MP_type','fuel_type','zone','date','hour','MW_forecast']

    data = pd.DataFrame(columns=columns)

    i = 0

    for filename in os.listdir('./VGForecastSummary'):
        i += 1
        if "_v" not in filename: continue
        print(filename)
        tree = ET.parse('./VGForecastSummary/'+filename)
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
    data['DateTime'] = data['date']+' ' + data['hour'].astype(str) + ':00'
    data['update_time'] = pd.to_datetime(data['update_time'])
    data['update_time'] = data['update_time'].dt.ceil('h')
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['PD_hours_back'] = (data['DateTime'] - data['update_time']).dt.components.hours
    data.to_csv('VGForecasts.csv')


def get_gen_output():
    i = 10
    columns = 'date,hour,fuel,generator,field,value'.split(',')
    data = pd.DataFrame(columns=columns)

    for filename in os.listdir('./GenOutputCapability'):

        if "_v" in filename: continue
        if filename == 'PUB_GenOutputCapability_20181017.xml': i = 0
        if i > 5: continue                                          #GET RID OF THIS LINE to do all files
        i += 1
        print(filename)
        tree = ET.parse('./GenOutputCapability/' + filename)
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
    data.to_csv('GenOutputCapability.csv')

if __name__ == "__main__":
    get_gen_output()
    #get_forecasts()