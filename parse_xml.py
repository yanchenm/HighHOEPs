import xml.etree.ElementTree as ET
import pandas as pd
import os


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

