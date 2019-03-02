import pandas as pd

fertility_rate = pd.read_csv('./data_sets/Fertility_Rate_to_Merge.csv')
endangered_languages = pd.read_csv('./data_sets/Endangered_Languages_Clean.csv')
english_proficiency = pd.read_csv('./data_sets/2017_EF_English_Proficiency.csv')
country_codes = pd.read_csv('./data_sets/country_codes.csv')

together = pd.merge(endangered_languages, fertility_rate, how='inner')
together_cleaner = together.drop(['Unnamed: 0'], axis=1)

country_codes_cleaner = country_codes.drop(['Unnamed: 0', 'M49 code'], axis=1)
eng_proficiency_cleaner = english_proficiency.drop(['Unnamed: 0'], axis=1)

together2 = pd.merge(eng_proficiency_cleaner, country_codes_cleaner,
                     left_on='Country', right_on='Country or Area', how='inner')

together2_minus_dups = together2.drop(['Country or Area'], axis=1)
final = pd.merge(together_cleaner, together2_minus_dups,
                 left_on='Country Code', right_on='ISO-alpha3 code', how='outer')


endangered_lang_potential_final = final.drop(['Country', 'Country Name', 'ISO-alpha3 code'], axis=1)
endangered_lang_potential_final = endangered_lang_potential_final.dropna(axis=0, subset=['Language'])

endangered_lang_potential_final.to_csv('./data_sets/endangered_languages_potential_final.csv')
