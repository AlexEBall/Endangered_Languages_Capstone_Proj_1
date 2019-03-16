import pandas as pd

endangered_languages = pd.read_csv('./data_sets/endangered_languages_final.csv', index_col=0)

# Functions for getting the dataset ready for ML
def create_countries_list(df):
    """ Creates a list of all countries where endangered langauges are spoken """
    countries_with_1_entry = list(
        df[df['Number of Countries Spoken'] == 1]['Countries Where Spoken'])
    countries_with_more_than_1_entry = list(
        df[df['Number of Countries Spoken'] >= 2]['Countries Where Spoken'])

    countries_with_more_than_1_entry = [
        x.split(',') for x in countries_with_more_than_1_entry]
    countries_with_more_than_1_entry = [
        item for sublist in countries_with_more_than_1_entry for item in sublist]

    all_countries = countries_with_1_entry + countries_with_more_than_1_entry
    all_countries = [x.strip() for x in all_countries]
    all_countries = list(set(all_countries))
    return all_countries


def add_one_hot_encoding_columns(countries_list, df):
    """ Creates columns from countries where the endangered langagues are spoken for one hot encoding """
    for country in countries_list:
        df[country] = 0


def one_hot_encode(df):
    """ One hot encode columns """
    countries_list = df['Countries Where Spoken']
    countries_list = countries_list.split(',')
    countries_list = [x.strip() for x in countries_list]

    for country in countries_list:
        df[country] = 1
    return df


countries_list = create_countries_list(endangered_languages)
add_one_hot_encoding_columns(countries_list, endangered_languages)
endangered_languages = endangered_languages.apply(one_hot_encode, axis=1)

endangered_languages = endangered_languages.drop(
    ['Latitude', 'Longitude', '2018 Band'], axis=1)

# Hot encode this one categorical column
endangered_languages = pd.concat([endangered_languages,
                                  pd.get_dummies(endangered_languages['Degree of Endangerment'])], axis=1)

endangered_languages = endangered_languages.drop(
    ['Degree of Endangerment', 'Countries Where Spoken', 'Country Code'], axis=1)

# Lastly convert the scientific notation to integers and drop the NaN values
endangered_languages['GDP Average (Current US $)'] = endangered_languages['GDP Average (Current US $)'].astype(int)
endangered_languages = endangered_languages.dropna()

endangered_languages.to_csv('./data_sets/endangered_languages_ML.csv', index=True)
