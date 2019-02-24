import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/EF_English_Proficiency_Index'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)

data = []
table = soup.find('table', attrs={'class': 'wikitable sortable'})

header_cols = table.find_all('th')
header_cols = [ele.text.strip() for ele in header_cols]
data.append([ele for ele in header_cols if ele])

table_body = table.find('tbody')
rows = table_body.find_all('tr')

for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])

# Have to drop the 2nd element because it comes up blank in this table's loop
del data[1]

# Extract the headers
headers = data.pop(0)

df = pd.DataFrame(data, columns=headers)

# Saving some memory
df['2018 Band'] = df['2018 Band'].astype('category')
df['2018 Rank'] = pd.to_numeric(df['2018 Rank'], errors='coerce')
df['2018 Score'] = pd.to_numeric(df['2018 Score'], errors='coerce')

df.to_csv('./data_sets/2017_EF_English_Proficiency.csv')

