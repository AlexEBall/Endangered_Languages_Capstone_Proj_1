# putting stuff here to hold
degree = together.groupby('Degree of Endangerment')
severely_endangered = degree.get_group('Severely endangered')

severely_endangered.plot(kind='scatter', y='Speakers', x='Fertility Rate Avg')
plt.show()


eng_speaking_countries = eng_speaking_countries.fillna(
    {'2018 Rank': 1.0, '2018 Score': 100.0, '2018 Band': 'Native Speakers'})

gdp['GDP Average (Current US $)'] = gdp['GDP Average (Current US $)'].fillna(0.0).astype(int)
gdp['GDP Average (Current US $)'] = [ millify(x) for x in gdp['GDP Average (Current US $)']]

# Frome EDA charts, etc
missing_rank = endangered_languages[endangered_languages['2018 Rank'].isnull()]
missing_rank['Country Code'].unique()
import plotly.plotly as py
import cufflinks as cf

%matplotlib inline
import plotly.graph_objs as go

speakers = endangered_languages[endangered_languages['Speakers'].notnull()]
# data = [go.Bar(
#             x=speakers['Language'],
#             y=speakers['Speakers']
#     )]
# py.iplot(data, filename='basic-bar')
speakers.index.values

sns.lmplot(x='Speakers', y='GDP Average (Current US $)', data=speakers)
# dropping null value
endangered_languages_complete = endangered_languages.dropna()

endangered_languages_complete.info()

ax = sns.barplot(x='Degree of Endangerment', y='2018 Rank',
                 data=endangered_languages_complete)

italy = endangered_languages_complete.groupby(
    endangered_languages_complete['Country Code']).get_group('ITA')
it = sns.boxplot(x='Degree of Endangerment',
                 y='Number of Countries Spoken', data=italy)

import matplotlib.pyplot as plt
endangered_languages_complete.plot(kind='hist', y='Speakers', bins=50)
plt.show()

endangered_languages_complete.plot(x='Degree of Endangerment')
plt.show()

endangered_languages_complete.tail(6)

italy[['Number of Countries Spoken']].plot(kind='bar')
high_proficiency = endangered_languages_complete.groupby(
    ['2018 Band']).get_group('High Proficiency')

high_proficiency.head()

high_proficiency.info()

endangered_languages_complete.hist('Speakers', bins=5, range=[0, 10])

endangered_languages_complete[endangered_languages_complete['Speakers'] < 2].count()

# Gaussian Kernel Density Estimate (KDE)

ax = sns.distplot(endangered_languages_complete['2018 Score'])

# Display pandas histogram (how pandas plots a hist)
# df['Award_Amount'].plot.hist()
# plt.show()

ax = sns.distplot(
    endangered_languages_complete['2018 Score'], hist=True, rug=True, kde_kws={'shade': True})

# Distributions/histograms are used for univariate analysis(analysis of one variable), however regression 
# analysis is used for bivariate beacuse we are looking for relationships between two variables
high_proficiency_wo_outliers = high_proficiency[high_proficiency['Speakers'] < 250000]

ay = sns.regplot(x='Speakers', y='2018 Score',
                 data=high_proficiency_wo_outliers)

# regplot = lowlevel(like pandas hist) while lmplot = highlevel(seaborn distplot)
aw = sns.lmplot(x='Speakers', y='2018 Score', data=high_proficiency_wo_outliers)
# The use of hue and columns, plotting multiple graphs while changing one variable is called faceting.

aw = sns.lmplot(x='Speakers', y='2018 Score', data=high_proficiency_wo_outliers, col='Number of Countries Spoken',
                sharex=False, sharey=False)
axes = aw.axes
axes[0, 0].set_ylim(50, 70)
axes[0, 1].set_ylim(50, 70)

ac = sns.lmplot(x='2018 Score', y='Speakers', data=high_proficiency_wo_outliers,
                sharex=False, sharey=False)

axes = aw.axes
axes[0, 0].set_ylim(50, 200000)
axes[0, 1].set_ylim(50, 70)
sns.despine(left=True, bottom=True)

for style in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
    sns.set_style(style)
    sns.distplot(endangered_languages_complete['2018 Score'])
    plt.show()

    for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.show()


# adding to sns plots

fig, ax = plt.subplots()
sns.distplot(df['Tuition'], ax=ax)
ax.set(xlabel="Tuition 2013-14",
       ylabel="Distribution", xlim=(0, 50000),
       title="2013-14 Tuition and Fees Distribution")

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                               sharey=True, figsize=(7, 4))
sns.distplot(df['Tuition'], ax=ax0)
sns.distplot(df.query('State == "MN"')['Tuition'], ax=ax1)
ax1.set(xlabel="Tuition (MN)", xlim=(0, 70000))
ax1.axvline(x=20000, label='My Budget', linestyle='--')
ax1.legend()


# USA map
# import plotly.graph_objs as go

df = pd.read_csv('./data_sets/2014_us_cities.csv')
df.head()

df['text'] = df['name'] + '<br>Population ' + \
    (df['pop']/1e6).astype(str)+' million'
limits = [(0, 2), (3, 10), (11, 20), (21, 50), (50, 3000)]
colors = ["rgb(0,116,217)", "rgb(255,65,54)", "rgb(133,20,75)",
          "rgb(255,133,27)", "lightgrey"]
cities = []
scale = 5000

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    city = go.Scattergeo(
        locationmode='USA-states',
        lon=df_sub['lon'],
        lat=df_sub['lat'],
        text=df_sub['text'],
        marker=go.scattergeo.Marker(
            size=df_sub['pop']/scale,
            color=colors[i],
            line=go.scattergeo.marker.Line(
                width=0.5, color='rgb(40,40,40)'
            ),
            sizemode='area'
        ),
        name='{0} - {1}'.format(lim[0], lim[1]))
    cities.append(city)

layout = go.Layout(
    title=go.layout.Title(
        text='2014 US city populations<br>(Click legend to toggle traces)'
    ),
    showlegend=True,
    geo=go.layout.Geo(
        scope='usa',
        projection=go.layout.geo.Projection(
            type='albers usa'
        ),
        showland=True,
        landcolor='rgb(217, 217, 217)',
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)"
    )
)

fig = go.Figure(data=cities, layout=layout)
# this will plot to your plotly account
# py.iplot(fig, filename='d3-bubble-map-populations')

# HEAT MAP
import plotly.graph_objs as go
import numpy as np

x = np.random.randn(2000)
y = np.random.randn(2000)

iplot([go.Histogram2dContour(x=x, y=y, contours=dict(coloring='heatmap')),
      go.Scatter(x=x, y=y, mode='markers', marker=dict(color='white', size=3, opacity=0.3))], show_link=False)


from plotly.graph_objs import *
trace1 = {
    'lat': list(endangered_languages['Latitude']),
    'lon': list(endangered_languages['Longitude']),
    'marker': {
        'color': np.random.randn(1931), 
        'colorscale': 'Viridis',
        'size': list(endangered_languages['Speakers']),
        'sizemode': 'area',
        'sizeref': 5000
    },
    'text': list(endangered_languages.index.values),
    'type': 'scattergeo',
    'uid': '6cf3d0'
}

data = Data([trace1])
layout = {
    'autosize': True,
    'geo': {
        'projection': {'type': 'orthographic'},
        'showcountries': True,
        'showlakes': True,
        'showland': True,
        'showocean': True,
        'showrivers': True
    },
    'height': 741,
    'hovermode': 'closest',
    'title': 'Endangered Langagues around the World',
    'width': 900
}

fig = Figure(data=data, layout=layout)
iplot(fig)