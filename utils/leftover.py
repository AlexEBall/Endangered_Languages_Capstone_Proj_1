degree = together.groupby('Degree of Endangerment')
severely_endangered = degree.get_group('Severely endangered')

severely_endangered.plot(kind='scatter', y='Speakers', x='Fertility Rate Avg')
plt.show()
