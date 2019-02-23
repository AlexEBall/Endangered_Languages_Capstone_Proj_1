import pandas as pd

data = pd.ExcelFile('./data_sets/Endangered_Languages.xlsx')
endangered_languages = data.parse('Extended_Dataset')
necessary_data = ['Name in English', 'Countries', 'Country codes alpha 3', 'Degree of endangerment',
                  'Number of speakers', 'Latitude', 'Longitude']
endangered_languages = endangered_languages[necessary_data]

""" 
    The column 'Country codes alpha 3' are string values separated by commas. 
    I want to access just the main country that langague is spoken, so converting the string into
    a list and then returning just the first country
"""
endangered_languages['Country codes alpha 3'] = endangered_languages['Country codes alpha 3'].fillna(
    "None")
endangered_languages['Country codes alpha 3'] = [
    x.split(',')[0] for x in endangered_languages['Country codes alpha 3']]


""" Locating the one row in 'Country codes alpha 3' that has no country code """
noneLook = endangered_languages['Country codes alpha 3'] == "None"

""" 
    After some research using the coordinates, was able to identify the country. 
    https://en.wikipedia.org/wiki/Shinasha_language 
    Adding that information to the 'Country codes alpha 3' and 'Countries' columns
"""
endangered_languages.iloc[405, 1] = 'Ethiopia'
endangered_languages.iloc[405, 2] = 'ETH'

""" Saving some memory by converting columns to category """
endangered_languages['Degree of endangerment'] = endangered_languages['Degree of endangerment'].astype(
    'category')

""" TODO: Should I keep NaN data for number of speakers? """
# missing_speakers = endangered_languages['Number of speakers'].isnull()
# endangered_languages[missing_speakers]

""" TODO: Should I drop the 3 entries with no coordinate data? """
# missing_lat = endangered_languages['Latitude'].isnull()
# endangered_languages[missing_lat]

""" Renaming columns """
col_dict = {'Name in English': 'Language', 'Countries': 'Countries Where Spoken', 'Country codes alpha 3': 'Country Code',
            'Degree of endangerment': 'Degree of Endangerment', 'Number of speakers': 'Speakers', 'Latitude': 'Latitude',
            'Longitude': 'Longitude'}

endangered_languages.columns = [col_dict.get(x, x) for x in endangered_languages.columns]

endangered_languages.to_csv('./data_sets/Endangered_Languages_Clean.csv')
