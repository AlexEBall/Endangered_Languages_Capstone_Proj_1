import pandas as pd

fertility_rate_df = pd.read_excel('./data_sets/Fertility-Rate.xls')
needed_data = ['Country Name', 'Country Code',
               '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
               '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
               '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
               '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
               '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
               '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
fertility_rate_df = fertility_rate_df[needed_data]
fertility_rate_df.set_index('Country Name', inplace=True)

""" Finding the average fertility rate for all countries """
# Should I make a copy here?
# fertility_rate_df_average = fertility_rate_df.copy()
# fertility_rate_df_average['Fertility Rate Avg'] = fertility_rate_df_average.mean(numeric_only=True, axis=1)
fertility_rate_df['Fertility Rate Avg'] = fertility_rate_df.mean(
    numeric_only=True, axis=1)

""" Find and drop rows that don't have a fertility rate mean value """
missing_mean = fertility_rate_df['Fertility Rate Avg'].isnull()
not_missing_mean = fertility_rate_df['Fertility Rate Avg'].notnull()
fertility_rate_df = fertility_rate_df[not_missing_mean]

""" Create another data set to merge with endangered languages """
fr_to_merge = fertility_rate_df[['Country Code', 'Fertility Rate Avg']]

fertility_rate_df.to_csv('./data_sets/Fertility_Rate_Clean.csv')
fr_to_merge.to_csv('./data_sets/Fertility_Rate_to_Merge.csv')
