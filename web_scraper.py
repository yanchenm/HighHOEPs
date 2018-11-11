import requests


def scrape_xml(path):
    url = path
    r = requests.get(url, allow_redirects=True)

    file_name = path.split('/')[-1]

    with open('data/xml/{}'.format(file_name), 'wb') as file:
        file.write(r.content)
