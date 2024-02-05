import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

from scipy.stats import linregress
from sklearn.metrics import r2_score

#Gather more insight with dataset using ydata profiling

#Import csv and generate report html
df = pd.read_csv("movie_dataset.csv")
profile = ProfileReport(df, title = "Profiling Report")
profile.to_file("your_report.html")


#df information
print(df.info()) #---> look at amount of NaNs in dataset and whether data types are correct
print(df.describe())

#Fixing column names that contain spaces
df = df.rename(columns={'Runtime (Minutes)': 'Runtime_(Minutes)', 
                        'Revenue (Millions)': 'Revenue_(Millions)'})

#Counting NaNs in Revenue and Metascore columns (We can also see the number of missing values in ydata profiling)
imdb_non_null_sum_rev = df['Revenue_(Millions)'].isna().sum()
print("NaNs in Revenue:", imdb_non_null_sum_rev)

imdb_non_null_sum_met = df['Metascore'].isna().sum()
print("\nNaNs in Metascore:", imdb_non_null_sum_met)


"""
From using the info() function there seems to be NaN values in Revenue and Metascore only.
info() also shows that data types are correct

If possible I want to fill the missing values if there is a high correlation between other numerical values.
If there is no way to fill these values then they will be dropped with dropna() 

From ydata profiling Rating and Metascore along with Revenue and Votes seem to be highly correlated from the report
Using pairplot from seaborn will also indicate where correlations lie to be further investigated
We might be able to fill in these values somewhat depending if we get high correlation values
"""

#Ploting with sns pairplot
sns.pairplot(df, kind = 'reg', plot_kws={'line_kws':{'color':'red'}})

"""
Pairplots visually confirm relationships between Votes and Revenue as well as Rating and Metascore
"""

#Dropping NaNs to compute regressions
df_drop = df.dropna()


#Scatter plot for Rating and Metascore with regression
plt.scatter(df_drop["Rating"], df_drop["Metascore"], c='blue')
plt.xlabel('Rating')
plt.ylabel('Metascore')
plt.title('Correlation between Rating and Metascore')
slope_1, intercept_1, r_value_1, _, _ = linregress(df_drop["Rating"], df_drop["Metascore"])
line_1 = slope_1*df_drop["Rating"]+intercept_1
plt.plot(df_drop["Rating"], line_1, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Rating and Metascore: {:.2f}".format(r_value_1**2))



#Scatter plot for Votes and Revenue
plt.scatter(df_drop["Votes"], df_drop["Revenue_(Millions)"], c='blue')
plt.xlabel('Votes')
plt.ylabel('Revenue_(Millions)')
plt.title('Correlation between Votes and Revenue_(Millions)')
slope_2, intercept_2, r_value_2, _, _ = linregress(df_drop["Votes"], df_drop["Revenue_(Millions)"])
line_2 = slope_2*df_drop["Votes"]+intercept_2
plt.plot(df_drop["Votes"], line_2, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Votes and Revenue: {:.2f}".format(r_value_2**2))


"""
Regression between Rating and Metascore: 0.45
Regression between Votes and Revenue: 0.41

They are correlated somewhat and we will fill the missing values using the straight lines fitted from the code above
"""

#Straght line function for rating and metascore
print("Straight line function for rating and metascore: y = {:.2f}x + {:.2f}".format(slope_1, intercept_1))


#Identify NaNs in the Metascore column
nan_indices_meta = df['Metascore'].isna()
df.loc[nan_indices_meta, 'Metascore'] = slope_1 * df.loc[nan_indices_meta, 'Rating'] + intercept_1

#Changing float to nearest real number value
df['Metascore'] = df['Metascore'].round().astype(int)


#Straght line function for votes and revenue
print("\nStraight line function for votes and revenue: y = {:.5f}x + {:.2f}".format(slope_2, intercept_2))

#Identify NaNs in the Revenue column
nan_indices_meta = df['Revenue_(Millions)'].isna()
df.loc[nan_indices_meta, 'Revenue_(Millions)'] = slope_2 * df.loc[nan_indices_meta, 'Votes'] + intercept_2

#Changing float to nearest real number value
df['Revenue_(Millions)'] = df['Revenue_(Millions)'].round().astype(int)


#Setting dropped na df to df clean
df_clean = df

#Question 1

#Find row index with the highest value for the Rating column
max_rating_index = df_clean['Rating'].idxmax()

#Retrieve the title with the maximum rating
max_rating_title = df_clean.loc[max_rating_index, 'Title']
# Print the result

print("The title with the highest rating is:", max_rating_title)


#Question 2

#Calculate total from 
sum_rev = sum(df_clean["Revenue_(Millions)"])

#Calculate the length of the cleaned dataframe
len_rev = len(df_clean)

#Calculate average
avg_rev = sum_rev/len_rev

print("The average revenue in million:", avg_rev)

#We can also obtain the average mean from describe()
print(df_clean.describe())

#or

avg_rev02 = df_clean["Revenue_(Millions)"].mean()

#Question 3

#Filter the dataframe to show rows for when the year column is between and including 2015 and 2017
df_fil_2015_2017 = df_clean[(df_clean['Year'] >= 2015) & (df_clean['Year'] <= 2017)]

# Calculate the average revenue from the filtered dataframe
avg_rev_2015_2017 = df_fil_2015_2017['Revenue_(Millions)'].mean()

print("The average revenue of movies from 2015 to 2017 is:", avg_rev_2015_2017)

#Question 4

#Use groupby and size functions to obtain the number of elements for each year and
#finally using get function to obtain the value 
movies_in_2016 = df_clean.groupby("Year").size().get(2016)

print("Number of movies released in 2016:", movies_in_2016)

#Question 5

#Doing the sum of 

#Calculate the sum of column directer where the string for "Christopher Nolan" is true
movies_by_nolan = (df_clean['Director'] == 'Christopher Nolan').sum()

print("The number of moveies directed by Christopher Nolan is:", movies_by_nolan)

#Question 6

#Filtering the rating column in the dataframe for values equal to or greater than 8
#and counting the occurances
df_fil_8_rating = df_clean["Rating"][(df_clean['Rating'] >= 8.0)].count()


print("The number of movies that have a rating of at least 8:", df_fil_8_rating)

#Question 7

#Filter array to only show rows where the director is Christopher Nolan
df_nolan = df_clean[df_clean['Director'] == 'Christopher Nolan']

#Calculate the median
median_rating_nolan = df_nolan['Rating'].median()

print("The median rating of movies directed by Christopher Nolan is:", median_rating_nolan)


#Question 8

#Group by the year and calculate the average rating
year_avg_rating = df_clean.groupby("Year")["Rating"].mean()

#Using idmax function to obtain id with maximum value
year_higest_avg_rating = year_avg_rating.idxmax()

print("The year with the highest average rating is:", year_higest_avg_rating)

#Question 9

#Similarly to question 4
count_2016 = df_clean.groupby("Year").size().get(2016)
count_2006 = df_clean.groupby("Year").size().get(2006)

#Calculating the percentage increase
per_increase = (count_2016-count_2006)/count_2006*100

print("Percentage increase of movies between 2006 and 2016:", per_increase)

#Question 10
#Generating a list containing all the actors from the df by splitting words with str.split
#that are seperated by a comma. The list of strings are then placed into seperate row using
#explode. Finally str.strip() is used to remove extra spaces 

actors = df['Actors'].str.split(',').explode().str.strip()
most_common_actor = actors.mode()[0]

print(f"The most common actor in all movies is: {most_common_actor}")


#Question 11

#Generating a list containing unique Genres from the df by splitting words with str.split
#that are seperated by a comma. The list of strings are then placed into seperate row using
#explode. Finally using the unique function to return unique strings

genre_words = df_clean['Genre'].str.split(',').explode().unique()

print("Amount of Genres in the dataset:", len(genre_words))

#Question 12 

#Evaluate relationships using pairplots 
sns.pairplot(df_clean, kind = 'reg', plot_kws={'line_kws':{'color':'red'}})


#Scatter plot for Rating and Metascore with regression
plt.scatter(df_clean["Rating"], df_clean["Metascore"], c='blue')
plt.xlabel('Rating')
plt.ylabel('Metascore')
plt.title('Correlation between Rating and Metascore')
slope_1, intercept_1, r_value_1, _, _ = linregress(df_clean["Rating"], df_clean["Metascore"])
line_1 = slope_1*df_clean["Rating"]+intercept_1
plt.plot(df_clean["Rating"], line_1, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Rating and Metascore: {:.2f}".format(r_value_1**2))


#Scatter plot for Votes and Revenue
plt.scatter(df_clean["Votes"], df_clean["Revenue_(Millions)"], c='blue')
plt.xlabel('Votes')
plt.ylabel('Revenue_(Millions)')
plt.title('Correlation between Votes and Revenue_(Millions)')
slope_2, intercept_2, r_value_2, _, _ = linregress(df_clean["Votes"], df_clean["Revenue_(Millions)"])
line_2 = slope_2*df_clean["Votes"]+intercept_2
plt.plot(df_clean["Votes"], line_2, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Votes and Revenue: {:.2f}".format(r_value_2**2))

#Scatter plot for rating and rank
plt.scatter(df_clean["Rating"], df_clean["Rank"], c='blue')
plt.xlabel('Rating')
plt.ylabel('Rank')
plt.title('Correlation between Rating and Rank')
slope_3, intercept_3, r_value_3, _, _ = linregress(df_clean["Rating"], df_clean["Rank"])
line_3 = slope_3*df_clean["Rating"]+intercept_3
plt.plot(df_clean["Rating"], line_3, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Votes and Revenue: {:.2f}".format(r_value_3**2))

#Scatter plot for rating and runtime
plt.scatter(df_clean["Rating"], df_clean["Runtime_(Minutes)"], c='blue')
plt.xlabel('Rating')
plt.ylabel('Runtime_(Minutes)')
plt.title('Correlation between Rating and Runtime_(Minutes)')
slope_4, intercept_4, r_value_4, _, _ = linregress(df_clean["Rating"], df_clean["Runtime_(Minutes)"])
line_4 = slope_4*df_clean["Rating"]+intercept_4
plt.plot(df_clean["Rating"], line_4, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Votes and Revenue: {:.2f}".format(r_value_4**2))

#Scatter plot for votes and runtime
plt.scatter(df_clean["Votes"], df_clean["Runtime_(Minutes)"], c='blue')
plt.xlabel('Votes')
plt.ylabel('Runtime_(Minutes)')
plt.title('Correlation between Votes and Runtime_(Minutes)')
slope_5, intercept_5, r_value_5, _, _ = linregress(df_clean["Votes"], df_clean["Runtime_(Minutes)"])
line_5 = slope_5*df_clean["Votes"]+intercept_5
plt.plot(df_clean["Votes"], line_5, c='red', label='Best-fit line')
plt.legend()
print("R-squared value between Votes and Revenue: {:.2f}".format(r_value_5**2))
