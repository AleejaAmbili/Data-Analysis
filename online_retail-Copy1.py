#!/usr/bin/env python
# coding: utf-8

# # Portfolio Project: Online Retail Exploratory Data Analysis with Python

# ## Overview
# 
# In this project, you will step into the shoes of an entry-level data analyst at an online retail company, helping interpret real-world data to help make a key business decision.

# ## Case Study
# In this project, you will be working with transactional data from an online retail store. The dataset contains information about customer purchases, including product details, quantities, prices, and timestamps. Your task is to explore and analyze this dataset to gain insights into the store's sales trends, customer behavior, and popular products. 
# 
# By conducting exploratory data analysis, you will identify patterns, outliers, and correlations in the data, allowing you to make data-driven decisions and recommendations to optimize the store's operations and improve customer satisfaction. Through visualizations and statistical analysis, you will uncover key trends, such as the busiest sales months, best-selling products, and the store's most valuable customers. Ultimately, this project aims to provide actionable insights that can drive strategic business decisions and enhance the store's overall performance in the competitive online retail market.
# 
# ## Prerequisites
# 
# Before starting this project, you should have some basic knowledge of Python programming and Pandas. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - numpy
# - seaborn
# - matplotlib
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`

# ## Project Objectives
# 1. Describe data to answer key questions to uncover insights
# 2. Gain valuable insights that will help improve online retail performance
# 3. Provide analytic insights and data-driven recommendations

# ## Dataset
# 
# The dataset you will be working with is the "Online Retail" dataset. It contains transactional data of an online retail store from 2010 to 2011. The dataset is available as a .xlsx file named `Online Retail.xlsx`. This data file is already included in the Coursera Jupyter Notebook environment, however if you are working off-platform it can also be downloaded [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx).
# 
# The dataset contains the following columns:
# 
# - InvoiceNo: Invoice number of the transaction
# - StockCode: Unique code of the product
# - Description: Description of the product
# - Quantity: Quantity of the product in the transaction
# - InvoiceDate: Date and time of the transaction
# - UnitPrice: Unit price of the product
# - CustomerID: Unique identifier of the customer
# - Country: Country where the transaction occurred

# ## Tasks
# 
# You may explore this dataset in any way you would like - however if you'd like some help getting started, here are a few ideas:
# 
# 1. Load the dataset into a Pandas DataFrame and display the first few rows to get an overview of the data.
# 2. Perform data cleaning by handling missing values, if any, and removing any redundant or unnecessary columns.
# 3. Explore the basic statistics of the dataset, including measures of central tendency and dispersion.
# 4. Perform data visualization to gain insights into the dataset. Generate appropriate plots, such as histograms, scatter plots, or bar plots, to visualize different aspects of the data.
# 5. Analyze the sales trends over time. Identify the busiest months and days of the week in terms of sales.
# 6. Explore the top-selling products and countries based on the quantity sold.
# 7. Identify any outliers or anomalies in the dataset and discuss their potential impact on the analysis.
# 8. Draw conclusions and summarize your findings from the exploratory data analysis.

# ## Task 1: Load the Data

# In[32]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


#Loading the data

file_path = "Online Retail.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls,sheet_name='Online Retail')


# In[34]:


# Cleaning the data

df.isnull().sum()  # check missing values
df = df.dropna(subset=['CustomerID'])
df = df.drop_duplicates()  # Removes duplicates
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df.head()


# In[35]:


#basic statistics of the dataset

df[['Quantity','UnitPrice']].describe()


# In[36]:


df['Quantity'].var(), df['UnitPrice'].var()


# In[41]:


print(df['Quantity'].describe())


# In[51]:


#data visualization to gain insights into the dataset.
# Histogram- shows how frequently value appears
#distribution of Quantity

plt.figure(figsize=(8,5))
sns.distplot(df[df['Quantity'] < 500]['Quantity'], kde=True, bins=50)
plt.xlabel('Quantity',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Distribution of Quantity')
plt.show()


# The graph shows right skewed data or positive skewed data which means a few small large values and more small values.
# In this case this means most orders are small but there are few high values.
# 

# In[56]:


#distribution of Quantity

plt.figure(figsize=(8,5))
sns.distplot(df[df['UnitPrice'] < 200]['UnitPrice'], kde=True, bins=30, color='green')
plt.xlabel('UnitPrice',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Distribution of UnitPrice')
plt.show()


# The graph shows right skewed data or positive skewed data which means a few small large values and more small values.
# In this case this means most of the products have low prices but there are few with high values.

# In[57]:


# Finding out top 10 most sold products and plotting using bar graph
# Top 10 products

top_products =df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print(top_products)

#bargraph

plt.figure(figsize=(10,5))
sns.barplot(x= top_products.values,y=top_products.index)
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Description')
plt.title('Top 10 Most Sold Products')
plt.show()


# In[60]:


# Finding out 10 least sold products and plotting using bar graph
# Least sold 10 products

top_products =df.groupby('Description')['Quantity'].sum().sort_values(ascending=True).head(10)
print(top_products)

#bargraph

plt.figure(figsize=(10,5))
sns.barplot(x= top_products.values,y=top_products.index)
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Description')
plt.title('Top 10 Least Sold Products')
plt.show()


# In[62]:


# Finding out top 10 revenue generating products and plotting using bar graph

#First we have to find the total revenue

df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']


# Top 10 products

top_Revenue_products =df.groupby('Description')['TotalRevenue'].sum().sort_values(ascending=False).head(10)
print(top_Revenue_products)

#bargraph

plt.figure(figsize=(10,5))
sns.barplot(x= top_Revenue_products.values,y=top_Revenue_products.index)
plt.xlabel('Total Revenue')
plt.ylabel('Product Description')
plt.title('Top 10 Most Revenue Generating Products')
plt.show()


# In[64]:


# Finding out 10 least revunue generating products and plotting using bar graph


# least revenue generating 10 products

Least_Revenue_products =df.groupby('Description')['TotalRevenue'].sum().sort_values(ascending=True).head(10)
print(Least_Revenue_products)

#bargraph

plt.figure(figsize=(10,5))
sns.barplot(x= Least_Revenue_products.values,y=Least_Revenue_products.index)
plt.xlabel('Total Revenue')
plt.ylabel('Product Description')
plt.title('Top 10 Least Revenue Generating Products')
plt.show()


# In[69]:


# Finding out top 10 most country by sales and plotting using bar graph
# Top 10 country

top_Countries =df.groupby('Country')['Quantity'].sum().sort_values(ascending=False).head(10)
print(top_Countries)

#bargraph

plt.figure(figsize=(10,5))
sns.barplot(x=top_Countries.index, y=top_Countries.values)
plt.xlabel('Country')
plt.ylabel('Total Quantity Sold')
plt.title('Top 10 Countries by Sales')
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.show()


# In[71]:


# Finding out last 10 most country by sales and plotting using bar graph
# Last 10 country

Last_Countries =df.groupby('Country')['Quantity'].sum().sort_values(ascending=True).head(10)
print(Last_Countries)

#bargraph

plt.figure(figsize=(10,5))
sns.barplot(x=Last_Countries.index, y=Last_Countries.values)
plt.xlabel('Country')
plt.ylabel('Total Quantity Sold')
plt.title('Last 10 Countries by Sales')
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.show()


# In[92]:


# Analysing Sales over time
# Monthly sales trend

#Extracting month and day of the week from data

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df['DayofWeek'] = df['InvoiceDate'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['DayofWeek'] = pd.Categorical(df['DayofWeek'], categories=weekday_order, ordered=True)


# items sold per month

Monthly_Sale = df.groupby('Month')['Quantity'].sum().sort_index()
print(Monthly_Sale)

#linergraph

plt.figure(figsize=(10,5))
sns.lineplot(x= Monthly_Sale.index,y=Monthly_Sale.values)
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.title('Monthly Sale Trend')
plt.xticks(range(1,13))
plt.grid(True)
plt.show()


# The sales seems to fluctuate during the first half of the year. After that there is a gradual growth till November, which marks the highest. After that there is slight drop in the sales.

# In[94]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract day names
df['DayofWeek'] = df['InvoiceDate'].dt.day_name()

# Define correct order of weekdays
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Convert DayofWeek into a categorical type with correct order
df['DayofWeek'] = pd.Categorical(df['DayofWeek'], categories=weekday_order, ordered=True)

# Group and sort data
Weekly_Sale = df.groupby('DayofWeek')['Quantity'].sum()

# Print weekly sales data
print(Weekly_Sale)

# Bar chart (better for categorical data)
plt.figure(figsize=(10,5))
sns.barplot(x=Weekly_Sale.index, y=Weekly_Sale.values, palette="viridis")

plt.xlabel('Day of Week')
plt.ylabel('Total Quantity Sold')
plt.title('Weekly Sales Trend')
plt.xticks(rotation=45)  # Rotate labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Weekly sales is highest on Thursday and lowest on Sunday

# In[95]:


df[df['DayofWeek'] == 'Saturday']


# No sales on Saturday

# In[98]:


# Finding Outliers

# using Boxplot

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Quantity'])
plt.title('Boxplot of Quantity')
plt.show()


# The points at the far right of the boxplot are outliers.

# In[99]:


# Outliers when comparing two data
# uses Scatterplot

plt.scatter(df['Quantity'],df['UnitPrice'])
plt.xlabel('Quantity')
plt.ylabel('Unit Price')
plt.title('Scatter Plot of Quantity vs Unit Price')
plt.show()


# Here, the plots at the far end of X axis and Y axis are outliers
# 

# In[ ]:




