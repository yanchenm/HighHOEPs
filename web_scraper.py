import os
import re

import requests
from bs4 import BeautifulSoup

from datetime import datetime


base_url = 'http://reports.ieso.ca/public/'


def scrape(url, format, type):

    r = requests.get(url, allow_redirects=True)

    file_name = url.split('/')[-1]
    report_name = url.split('/')[-2]

    print('Downloading {}...'.format(url))

    file_path = './data/{}/{}/{}'.format(type, format, report_name)

    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    with open('./data/{}/{}/{}/{}'.format(type, format, report_name, file_name), 'wb') as file:
        file.write(r.content)


def scrape_report(report_name, data_type, start_date, end_date):
    url = base_url + report_name
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all('a')

    # Scrape csv reports if available
    if report_name in ['GenOutputCapability', 'VGForecastSummary']:
        file_type = 'xml'
    else:
        file_type = 'csv'

    report_root = './data/{}/{}/{}'.format(data_type, file_type, report_name)

    if not os.path.isdir(report_root):
        os.mkdir(report_root)

    # Reports desired differs for prediction vs. actual data
    if report_name in ['VGForecastSummary', 'PredispMktPrice', 'PredispMktTotals']:
        pattern = r'PUB_[a-zA-Z]+_([0-9]{{8}})[0-9]*_v[0-9]+\.{}'.format(file_type)
    else:
        pattern = r'PUB_[a-zA-Z]+_([0-9]{{8}})[0-9]*\.{}'.format(file_type)

    for link in links:
        # Only get links to reports
        if link.get_text() not in ['Name', 'Last modified', 'Size', 'Description', 'Parent Directory']:
            report_file_name = link['href']
            report_url = url + '/' + report_file_name

            match = re.match(pattern, report_file_name)

            if match is not None:

                date = match.group(1)
                date = datetime.strptime(date, '%Y%m%d')

                if start_date <= date <= end_date:
                    if not os.path.isfile('{}/{}'.format(report_root, report_file_name)):
                        scrape(report_url, file_type, data_type)


def scrape_all_reports(start_date, end_date, data_type):

    if not os.path.isdir('./data/'):
        os.mkdir('./data/')

    if not os.path.isdir('./data/{}'.format(data_type)):
        os.mkdir('./data/{}'.format(data_type))
        os.mkdir('./data/{}/csv'.format(data_type))
        os.mkdir('./data/{}/xml'.format(data_type))

    reports = ['DispUnconsHOEP', 'GenOutputCapability', 'PredispMktPrice', 'PredispMktTotals',
               'RealtimeMktPrice', 'RealtimeMktTotals', 'VGForecastSummary']

    for report in reports:
        print('Scraping {}...'.format(report))
        scrape_report(report, data_type, start_date, end_date)


if __name__ == '__main__':
    start_date = datetime(2018, 10, 13)
    end_date = datetime(2018, 11, 24)
    scrape_all_reports(start_date, end_date, 'data')
