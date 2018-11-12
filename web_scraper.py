import os
import re

import requests
from bs4 import BeautifulSoup


base_url = 'http://reports.ieso.ca/public/'


def scrape_xml(path):
    url = path
    r = requests.get(url, allow_redirects=True)

    file_name = path.split('/')[-1]
    report_name = path.split('/')[-2]

    print('Downloading {}...'.format(url))

    file_path = './data/xml/{}'.format(report_name)

    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    with open('./data/xml/{}/{}'.format(report_name, file_name), 'wb') as file:
        file.write(r.content)


def get_updated_time(path):
    pass


def scrape_historic(report_name):
    url = base_url + report_name
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        # Only get links to reports
        if link.get_text() not in ['Name', 'Last modified', 'Size', 'Description', 'Parent Directory']:
            report_file_name = link['href']
            report_url = url + '/' + report_file_name

            # Match only the most recent version of each report
            pattern = r'[a-zA-Z]+_[a-zA-Z]+_[0-9]+\.xml'

            if re.match(pattern, report_file_name) is not None:
                if not os.path.isfile('./data/xml/{}/{}'.format(report_name, report_file_name)):
                    scrape_xml(url + '/' + report_url)
