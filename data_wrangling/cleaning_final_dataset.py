import pandas as pd

endangered_languages = pd.read_csv(
    './data_sets/endangered_languages_potential_final.csv', index_col=0)
gdp_data = pd.read_csv('./data_sets/gdp_data_to_merge.csv')

# Noticed some descrepencies in the data, especially after mering the EF English scores
# I wanted to give native English speaking countires a score of 100, rank of 1 and category
# 'native speakers'

eng_speakers = endangered_languages[endangered_languages['Country Code'].isin(
    ['GBR', 'USA', 'CAN', 'AUS', 'NZL', 'IRL'])]

eng_speakers = eng_speakers.fillna(
    {'2018 Rank': 0.0, '2018 Score': 100.0, '2018 Band': 'Native Speakers'})
endangered_languages.update(eng_speakers)

# Another thing I noticed was that after mapping the country codes to EF data in merging_datasets.py
# Some values didn't map correctly and this was because of the difference in spelling of coutnries
# For example, in one data set Russia is used but in the other it was Russian Federation
# So I found the unique values of countries missing rank and then compared to EF data to manually update

missing_rank = endangered_languages['2018 Rank'].isnull()
endangered_languages[missing_rank]['Country Code'].unique()

# Should update Belarus country code to BLR (Belarus was given Barbados country code)
endangered_languages.loc['Belarusian', 'Country Code'] = 'BLR'
endangered_languages.loc['Polesian', 'Country Code'] = 'BLR'

# Russia has EF score
# South Korea has EF score
# Belarus, Iran, Vietnam, Syria
# Bolivia, Venezuela, Dominican (DMA) --> These need to be updated

russia = endangered_languages[endangered_languages['Country Code'] == 'RUS']
korea = endangered_languages[endangered_languages['Country Code'] == 'KOR']
belarus = endangered_languages[endangered_languages['Country Code'] == 'BLR']
iran = endangered_languages[endangered_languages['Country Code'] == 'IRN']
vietnam = endangered_languages[endangered_languages['Country Code'] == 'VNM']
syria = endangered_languages[endangered_languages['Country Code'] == 'SYR']
bolivia = endangered_languages[endangered_languages['Country Code'] == 'BOL']
venezuela = endangered_languages[endangered_languages['Country Code'] == 'VEN']
dominicana = endangered_languages[endangered_languages['Country Code'] == 'DMA']

russia = russia.fillna(
    {'2018 Rank': 42.0, '2018 Score': 52.96, '2018 Band': 'Moderate Proficiency'})
korea = korea.fillna({'2018 Rank': 31.0, '2018 Score': 56.27,
                      '2018 Band': 'Moderate Proficiency'})
belarus = belarus.fillna(
    {'2018 Rank': 38.0, '2018 Score': 53.53, '2018 Band': 'Moderate Proficiency'})
iran = iran.fillna({'2018 Rank': 66.0, '2018 Score': 48.29,
                    '2018 Band': 'Very Low Proficiency'})
vietnam = vietnam.fillna(
    {'2018 Rank': 41.0, '2018 Score': 53.12, '2018 Band': 'Moderate Proficiency'})
syria = syria.fillna({'2018 Rank': 76.0, '2018 Score': 46.37,
                      '2018 Band': 'Very Low Proficiency'})
bolivia = bolivia.fillna(
    {'2018 Rank': 61.0, '2018 Score': 48.87, '2018 Band': 'Low Proficiency'})
venezuela = venezuela.fillna(
    {'2018 Rank': 75.0, '2018 Score': 46.61, '2018 Band': 'Very Low Proficiency'})
dominicana = dominicana.fillna(
    {'2018 Rank': 37.0, '2018 Score': 54.97, '2018 Band': 'Moderate Proficiency'})


endangered_languages.update(russia)
endangered_languages.update(korea)
endangered_languages.update(belarus)
endangered_languages.update(iran)
endangered_languages.update(vietnam)
endangered_languages.update(syria)
endangered_languages.update(bolivia)
endangered_languages.update(venezuela)
endangered_languages.update(dominicana)

# Just to check everything worked out
endangered_languages[endangered_languages['Country Code'] == 'RUS'].head()

# Creating a new feature column to document how many countries that langauge is spoken in
endangered_languages['Number of Countries Spoken'] = [
    len(x.split(',')) for x in endangered_languages['Countries Where Spoken']]


# Modifiying '2018 Score with native speaking countries tied at rank 1, every other country bumped up by 1
endangered_languages['2018 Rank'] = [
    x + 1 for x in endangered_languages['2018 Rank']]

# resetting the index to merge cleanly
endangered_languages = endangered_languages.reset_index()

final = pd.merge(endangered_languages, gdp_data,
                 on='Country Code', how='inner')
final.set_index('Language')

final.to_csv('./data_sets/endangered_languages_final.csv', index=False)
