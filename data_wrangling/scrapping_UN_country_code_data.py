import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://unstats.un.org/unsd/methodology/m49/'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)

data = []
table = soup.find('table', attrs={'class': 'table table-striped'})

header_cols = table.find_all('th')
header_cols = [(ele.text.strip()).encode('utf-8') for ele in header_cols]
data.append([ele for ele in header_cols if ele])

rows = table.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [(ele.text.strip()).encode('utf-8') for ele in cols]

    data.append([ele for ele in cols if ele])

# Have to drop the 2nd element because it comes up blank in this table's loop
del data[1]

# Extract the headers
headers = data.pop(0)

df = pd.DataFrame(data, columns=headers)

df.drop([u'M49 code'], axis=1)

df.to_csv('./data_sets/country_codes.csv', index=False)
