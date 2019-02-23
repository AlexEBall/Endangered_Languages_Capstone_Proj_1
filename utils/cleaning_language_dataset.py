import pandas as pd

data = pd.ExcelFile('./data_sets/Endangered_Languages.xlsx')
endangered_languages = data.parse('Extended_Dataset')
necessary_data = ['Name in English', 'Countries', 'Country codes alpha 3', 'Degree of endangerment',
                  'Number of speakers', 'Latitude', 'Longitude']
endangered_languages = endangered_languages[necessary_data]

""" The column 'Country codes alpha 3' are string values separated by commas. 
    I want to access just the main country that langague is spoken, so converting the string into
    a list and then returning just the first country
"""
endangered_languages['Country codes alpha 3'] = endangered_languages['Country codes alpha 3'].fillna("None")
endangered_languages['Country codes alpha 3'] = [x.split(',')[0] for x in endangered_languages['Country codes alpha 3']]


""" Filling the missing values for 'Country codes alpha 3' then dropping the one instance of that """
noneLook = endangered_languages['Country codes alpha 3'] == "None"
endangered_languages.drop(endangered_languages.index(405))
