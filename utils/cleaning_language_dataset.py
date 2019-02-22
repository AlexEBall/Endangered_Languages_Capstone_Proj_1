import pandas as pd

data = pd.ExcelFile('./data_sets/Endangered_Languages.xlsx')
endangered_languages = data.parse('Extended_Dataset')
necessary_data = ['Name in English', 'Countries', 'Country codes alpha 3', 'Degree of endangerment',
                  'Number of speakers', 'Latitude', 'Longitude']

endangered_languages = endangered_languages[necessary_data]


