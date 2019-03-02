degree = together.groupby('Degree of Endangerment')
severely_endangered = degree.get_group('Severely endangered')

severely_endangered.plot(kind='scatter', y='Speakers', x='Fertility Rate Avg')
plt.show()


eng_speaking_countries = eng_speaking_countries.fillna(
    {'2018 Rank': 1.0, '2018 Score': 100.0, '2018 Band': 'Native Speakers'})
