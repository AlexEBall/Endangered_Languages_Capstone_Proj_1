<h1>Endangered Languages Capstone Project 1</h1>
<h3>Report</h3>

<img align="center" src="https://media.giphy.com/media/pzmbXFDiRbEEk1vCtP/giphy.gif">

<h2 align="center">Problem</h2>

In all seriousness, there are many endangered languages in the world today. Many that have already gone extinct. I would like to know what features cause a language to go extinct and to create a machine learning model to predict how likely an endangered language would go extinct. There are many variables to consider when dissecting this problem and I by no means will have a definitive answer once I complete my machine learning model. This problem may not even be a problem which can be solved by machine learning, nonetheless I’d like to state my hypothesis and go over how I wrangled and cleaned the dataset, how I applied some exploratory data analysis along with inferential statistics to further understand the problem. 

Here's a TLDR link to the [google slides presentation ](https://docs.google.com/presentation/d/1MxG_4JAO5tjp5d2ADxXUuIYpUNWR_-_YsInlD9pk8OA/edit?usp=sharing)

<h2 align="center">Hypothesis</h2>

Low fertility rate and GDP in the country of the language along with a proximity to English speaking centers cause a language to lose speakers which will eventually lead that language to becoming extinct. The first, low fertility rate means that there are fewer and fewer children being born which would mean less people to pass on the language from generation to generation. The second, a low GDP, would mean that, that country would have less resources available to build schools or centers which would preserve the language for future generations. And lastly, proximity to English speaking centers (centers here being defined as cities) would mean that speakers of an endangered language might begin to learn English for work opportunities and their children may go to English speaking schools, where they might be ridiculed for speaking another language. 

<h2 align="center">Collecting the Data</h2>

For this project I was able to find a fairly comprehensive dataset from data.world which included over 2500 rows documenting various endangered languages from around the globe. That was the first dataset I was going to include. The next dataset I worked on finding was fertility rate for countries, and luckily I found one that contained data from 1960 to 2013. The reason I wanted to track fertility rate is because one of my hypothesis is that the rate of children being born could potentially affect how many speakers of a language there could be. These first two datasets were already in .csv format so I just had to download them, however the next dataset was on the web. I scraped the dataset from Wikipedia (I could have used their API but decided to try my hand at scraping). The dataset looked at countries taking the ETF test, which is an English proficiency test and ranked countries on their level of proficiency . Another reason to include this dataset is because how much English is spoken and or valued in that country may affect how a native language is used, again another one of my hypothesis I aim to test using my ML model. Finally, again scraping though this time from UN data, I was able to get a table of country names and their corresponding alpha-3 codes. I want to continue using this dataset as a map to help merge data frames (more on that later).

<h2 align="center">Wrangling and Cleaning the Data</h2>

A lot of this data was in pretty bad shape. Especially the two .csv files I downloaded. For the first from data.world with the endangered languages, the first problem I came across was columns listing what countries the languages were spoken in. In fact there were two columns, one was alpha-3 codes and another country names, both however were stored as strings separated by commas. I decided to use list comprehension on the alpha-3 columns to slice the string and extract just the first (or main) country from the string. I made sure it was the main country because that dataset also had lat and long columns that when plotted on a map, showed the language centered in the first country from the alpha-3 column string. I wanted to have alpha-3 columns in all the datasets to make merging easier, and since the country name column contained the same values just in their long form, I didn’t feel to bad about losing trimming those strings. That dataset also had a bunch of NaN values that I ended up dropping seeing as they represented less than 10% of the data. My mentor, agreed that this would be the best way to handle this situation as backfilling, forward filling or applying the mean of the entire column would lead to biased results. The fertility rate dataset also was problematic because it contained a few countries with entire rows of NaN values for fertility rate. I ended up just dropping these rows because if they didn’t have any data, they wouldn’t be relative to solving my problem. I also had to run some aggregate functions across each row (country) to get the mean, median, max, min and other statistical values over the years of 1960 to 2013. Those statistical values would be much more valuable than just per year fertility rate per country. The two scraped datasets were much less problematic, in fact the one from Wikipedia was really clean and needed minimal processing but the one from the UN suffered from being encoded in unicode. That made the text a bit difficult to parse and I needed to decode it back to utf-8. Once all datasets were cleaned, I was able to merge them into one data frame that included all the features I wanted my model to test

Once all datasets were wrangled and cleaned I began to merge them into one dataframe. This posed its own issues as some values were mishandled ( I only noticed when taking a deeper dive). For instance, when mapping English proficiency scores from scraped data, I realized there were discrepancies in the way the datasets spelled country names. I had to go in latter and manually update values for each country. There were about 12 in total, I also had to do some other processing in which I gave some scores to English speaking countries, like the US, Canada, England, etc.. These countries didn’t take the English proficiency exam because they are native English speaking countries. After discussing with my mentor we came to the conclusion that giving them a rank of 1 (the highest possible) along with scores of 100 which represented a native level of understanding was the best approach here. 

All the cleaning and wrangling python scripts can be found [here](https://github.com/AlexEBall/Endangered_Languages_Capstone_Proj_1/tree/master/data_wrangling)

<h2 align="center">Exploratory Data Analysis and Inferential Statistics</h2>

The goal of this is to apply some inferential statistics to the my cleaned Endangered Languages dataset to find the variables that can help me answer the questions I'm posing. Along with finding out if there is any strong correlation between the data. And with finding the best tests to analyze the data.

I first wanted to get a sense of the distribution across level of endangeredness, so I plotted a stripplot to see how that turned out. 

<img align="center" src="https://user-images.githubusercontent.com/29084524/57589297-d234a680-74d6-11e9-9310-e112919cd82a.png">

As we would expect, vulnerable languages had the largest variance in terms of number of speakers, followed by definitely endangered, severely endangered, critically endangered and finally extinct languages all piling up at zero.

Next, I wanted to know which if any features in the dataset are correlated with the number of speakers of the languages. My first hunch was that the fertility rate for each country, the GDP of the country and that country’s overall English speaking ability would correlate very highly with the number of speakers. The first test I conducted was to look at the Pearson Correlation Coefficient which measures the linear correlation between two variables X and Y. I then plotted the findings in a heatmap through Python. 

<img align="center" src="https://user-images.githubusercontent.com/29084524/57589292-d19c1000-74d6-11e9-96f2-315c6b07a4e9.png">

My first findings were highly surprising as many of the variables had negative correlation to the number of speakers. I thought at first it was my method, so I plotted a regression line isolating the 2018 rank (which had a super low correlation rate to number of speakers) and number of speakers. I assumed that at first this would be much higher because a low rank would mean that these languages weren’t influenced by English speaking centers. The second order polynomial regression line shows that as the countries rank of English proficiency goes up, so to does the number of speakers until it reaches a critical mass where it begins to become negatively correlated, the less proficient the region is at speaking English, the more speakers of that language there are. 

This could be due to a number of reasons, for one, I’m looking at just English speaking centers which doesn’t account for areas where Spanish, Chinese or some other dominant language influences the native speakers of that language. 

Finally I thought, it may be worth applying a Chi Squared Test on some of these categorical columns. I retrieved my dataset before I had processed it for machine learning (meaning no hot encoded columns) and ran some tests on some of the variables, mainly Speakers and Fertility Rate Avg. I first broke down the languages into five categories based on their levels of endangerment and proceeded to make a count of fertility rates for that level (some languages would have the same fertility rate, as they were from the same country but different level of endangerment). Once that was done I posed a null hypothesis that there was no correlation between number of speakers for each level of endangerment and the fertility rate. After some calculations I found that the p-value was basically zero, which was under the pre-chosen alpha of 0.05. With a p-value < 0.05 , we can reject the null hypothesis. There is definitely some sort of relationship between 'Degree of endangerment' and the 'Fertility Rate Avg' column. We don't know what this relationship is, but we do know that these two variables are not independent of each other

<img align="center" src="https://user-images.githubusercontent.com/29084524/57589291-d19c1000-74d6-11e9-9e53-edd5a54583a4.png">

Again, this is an extremely complicated problem and one that machine learning might not be able to solve. The number of possible features to consider is vast and the true data isn’t readily available. I’ve made the best attempt with the data I could find and will continue to process this information and see if my model can’t learn some interesting insights from the data that I wasn’t able to glean. 

Links to the two EDA notebooks found [here](https://github.com/AlexEBall/Endangered_Languages_Capstone_Proj_1/tree/master/statistical_inferences)

<h2 align="center">In-Depth Analysis Using Machine Learning</h2>

The code to the full logistic regression model can be found [here](https://github.com/AlexEBall/Endangered_Languages_Capstone_Proj_1/blob/master/machine_learning_models/Logistic%20Regression%20Model.ipynb)

The problem I tried to tackle was to figure out why languages become extinct. Obviously this problem has many variables and as I wrangled and cleaned the data (from various datasets) I realized that this problem would be hard to translate into a machine learning algorithm. Nonetheless, I thought if I used best practices when preprocessing my data that I would find some interesting patterns or themes underlying the data. Firstly, each observation in the dataset contained a column that listed countries where that language was spoken. On advice from my mentor I hot-encoded that feature. He also advised that I hot-encode the column that assigned a level of endangerment to each language. Once that was completed, the data was ready for machine learning. 

I decided to use multiple logistic regression to tackle this problem because it’s one of the simpler models to use when doing classification. Also it tends to handle outliers better than a simple linear regression model because of the loggit function it uses whose inverse is the sigmoid equation. This is used as logistic regression’s cost function to readjust the weights as the model is fitted to the data. Let’s take a second to go into logistic regression a bit. As I said before, logistic regression is used for classification problems where independent variables are continuous in nature and the dependent variable is in categorical form, classes like a positive and negative class. For my dataset, that target variable was if a language was extinct or not. Extinction would be classified as 1 while non-extinct languages would be classified as 0. 

After importing all the necessary modules from pandas, numpy, seaborn and scikit learn. I wanted to get a count of my target variable classes and ran into an issue right from the start. 

<img src="https://user-images.githubusercontent.com/29084524/57589294-d19c1000-74d6-11e9-9d77-7a02e85d056e.png">

According the graph, my classes were highly imbalanced. Running logistic regression with this kind of imbalance would result in terrible performance metrics. 
	
This means as another step to the preprocessing step would be to balance the classes either through upsampling or downsampling. Upsampling involves resampling from the minority class (here being extinct languages) with replacement to match the number of samples in the majority class.

I upsampled and split, trained and tested my logistic regression model right out of the box. And the results didn’t bode too well. The model was only predicting the extinct classes and had an accuracy score of 51%. I thought that this may be because of one of upsampling traits, which is that the process of randomly duplicating observations from the minority class reinforces its signal. The signal here being the extinct class. To counter this effect, I decided instead to downsample the data, which is reducing the amount of the majority class (without replacement) to match the count of the minority class. 

Repeating the split, train and test process with logistic regression on the downsampled data yielded better results but not by much. The model’s accuracy went up to 57% but was still only predicting the extinction class. This left me thinking about what steps I could take to up the accuracy score. One of the key preprocessing steps I neglected was scaling the data. Many of my features varied widely among one another. For example the number of speakers ranged from 0 to several millions, while fertility rate would range from just above 0 to about 8. With these values being so radically different the summary statistics would also share these traits and since many machine learning algorithms work better when features are on a relatively similar scale and close to normally distributed I needed to scale this data. 

There are several different scalers available in the scikit learn API and each do scale the data in different ways. I opted with using Robust Scaler which transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value). I mainly used it because it minimizes the effect of outliers. So I scaled the data using Robust Scaler and then downsampled the data as before. Once the data was preprocessed I split, tested and trained my model and lo and behold, the accuracy jumped up to 99% and was now actually predicting both the extinct and non-extinct classes. 

```
log_reg = LogisticRegression(solver='lbfgs')

scaler = RobustScaler(quantile_range=(25, 75))
col_names = endangered_languages.columns

col_list = list(col_names)
col_list.pop(0)

X = endangered_languages.drop(['Language'], axis=1)

scaled_langauges = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_langauges, columns=col_list)

df_majority = scaled_df[scaled_df['Extinct'] == 0]
df_minority = scaled_df[scaled_df['Extinct'] == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=193,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

X = df_downsampled.drop(['Extinct'], axis=1)
y = df_downsampled['Extinct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# How many classes is our model predicting?
print('Clsses Predicted: ', np.unique(y_pred))

# How's our accuracy now?
print('Accuracty: ', accuracy_score(y_test, y_pred))
```

| | 1 or Positive Class | 0 or Negative Class |
|:---:|:---:|:---:|
| Extinct | 49 | 0 |
| Non Extinct | 1 | 66 |

Looking at the confusion matrix, out of the 49 extinct languages, the classifier correctly predicted all 49 of them. Out of the 67 non-extinct languages, the classifier correctly predicted 66 of them. The logistic regression model only mis-predicted 1 observation. 

<img src="https://user-images.githubusercontent.com/29084524/57589295-d234a680-74d6-11e9-996f-7c12b9926466.png">

These initial results seemed too good to be true. The ROC curve to the left looked like an example of what a perfect model would result in. A healthy dose of skepticism in me, lead to further investigation as I didn’t believe this to be a true result. There must be something I overlooked. 

I wanted to do some feature engineering because I believed that some of my features were overfitting my model. I dropped the speakers (an integer value of the number of speakers of that language) along with the hot-encoded columns designating that languages level of endangerment (i.e., Vulnerable, Definitely, Severely and Critically Endangered). The reasoning behind this is because I thought these had too high of a correlation to extinction. After dropping these features in my training data, I once again performed logistic regression and my model dropped back down to a 52% accuracy. This made sense to me especially for the hot-encoded columns, as the model might begin to think, if a language is neither vulnerable, definitely, severely nor critically endangered… it must be extinct. My original hypothesis was that a low fertility rate and overall GDP plus a high level of English speaking proficiency would cause a language to become extinct. I boiled down my dataset into four different data subsets and found a very interesting trend. I kept the fertility rate, GDP, English proficiency, number of speakers and number of countries that language is spoken in features for each of the 4 data subsets and rotated one of the level of endangerment hot-encoded columns. All of these logistic regression models were trained and tested with the robustly scaled and downsampled data. Each of the models predicted both the 0 and the 1 classes and these are their accuracy scores. 

| Level of Endangerment |  Accuracy Score |
|:---------------------:|:---------------:|
| Vulnerable | 73% |
| Definitely | 75% |
| Severely | 76% |
| Critically | 81% |

<img src="https://user-images.githubusercontent.com/29084524/57589296-d234a680-74d6-11e9-9901-35bbf9bb603e.png">

The final model, with the critically endangered column scored the best and its ROC curve (left) looked more digestible. However, what was interesting to me was the upward trend in accuracy. As the model learned about which observations (languages) were associated with various degrees of endangerment it performed better and better. This I would say is evidence for my hypothesis that the features of fertility rate, GDP and English proficiency scores do have some effect on whether or not a language will go extinct or not. 

I did some cross validation, with K-folds Cross Validation and hyperparameter tuning with GridSearch to find an optimized C for my logistic regression model. C here being the inverse regularization parameter, or basically a penalty term meant to disincentivize and regulate against overfitting. Once hypertuned and cross validated the model jumped up to a 93% accuracy score. 

<h2 align="center">Conclusion</h2>

What I learned through doing this machine learning analysis is that once preprocessing, scaling and feature engineering were done (which are all expensive to do) a simple model like logistic regression can yield good results. If I were to continue with this model, I would like to add some data of more healthy languages to the dataset to balance out the endangered ones. Since the dataset only contained endangered languages I feel as though the model became bias to that information even though it performed well. Again, some problems are not designed to be solved by machine learning and thinking about what factors contribute to the extinction of a language could be one of them but after performing my machine learning analysis I can more comfortably say that fertility rate, GDP and English proficiency do have some effect on a languages likelihood of extinction.

