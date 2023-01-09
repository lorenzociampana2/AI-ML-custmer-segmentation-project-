# Ai and ML final project
CUSTOMER SEGMENTATION PROJECT
Luiss, Aa. 2022/23

Lorenzo Ciampana 
Alberto Fornari
Alessio Attolico

INTRODUCTION:

Our firm wants to segment the customers in the Brazilian area in order to launch an email campaign which is as more precise and personalized for each type of customer as possible, so that they can target them based on their habits.
To do it we are going to use a very common business analysis method, which is the RFM:
-	Recency: time since a customer’s last purchase
-	Frequency: total number of purchases of a customer
-	Monetary value: total amount of money spent by a customer

CLUSTERING METHODS:

In order to perform a good clustering among customers, we decided to use the following methods:
-	K-means clustering: is one of the most  used clustering method, especially in this cases. We had the opportunity in fact to look at some similar business projects on the web where this algorithm is always used;
-	Hierarchical clustering: Hierarchical clustering is another popular method of grouping data. It creates groups so that data within a group are similar to each other and different from objects in other groups. Clusters are visually represented in a hierarchical tree called a dendrogram.
-	PCA (Principal Component Analysis): this method is very useful whenever we have to deal with large datasets as it helps us reducing the dimensionality of such datasets and minimizing the amount of useful information so that we can have a clearer and more precise interpretation of it.


PROJECT CORE:

Let’s now dive into the steps that we followed to have a clear visualization of the dataset, which can be divided as follows:
-	EDA (Exploratory Data Analysis):
This process is very important for every type of data analysis project as it allows us to have an overall overview of the data, to understand if there are any type of anomalies or null values or just to make some initial assumptions of the data frame. 
In our case it was very propaedeutic because the majority of the future steps won’t have been that precise without a proper data cleaning.
Here we have some examples of our EDA with some plots:

data.head()

 

Converting the variables into dates by using Pandas and then distincting date from time:

dates = ['order_purchase_timestamp', 'order_approved_at', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date']
for i in dates:
    data[i] = pd.to_datetime(data[i])

data['order_date'] = [d.date() for d in data['order_purchase_timestamp']]
data['order_time'] = [d.time() for d in data['order_purchase_timestamp']]



	data.describe() #dataset visualization with all the values

	data.columns

	data.nunique() #returns the output of number of unique values 

	data.isnull().sum() sum of all the null values of each variable

	Removing duplicates: 
	data.duplicated().sum()
data.drop_duplicates(keep='first', inplace=True)
data.reset_index(drop=True, inplace=True)
Outliers Analysis: 
analyzing some variables, we noticed that in some plots there are data points that exceed the average, for example, in ‘payment value’, which is in our mind one of the principal variable to study for a big firm, we can suppose that there are some customers who have paid an amount of money which is relatively higher than the others and this can affect our analysis in the future:

plt.figure() 
data.reset_index().plot(kind='scatter', x='index', y='payment_value', c='gray')

 
mean_pv = data["payment_value"].mean()  #result mean = 195
std_pv = data["payment_value"].std()    #result Standard Deviation = 295.5
print(mean_pv)
print(std_pv) 

results:
195.20144627496734 #mean
295.4593435200087 #std
Taking a look at the scatterplot of the outliers, we can notice that there are few payments above (around) 3500$, so we want as an output all the payments above that amount and then delete them:
outlier1 = data[data['payment_value'] > 3500]
print('\nOutlier dataframe:\n', outlier1)

#the output will be 5 payments, each one with all its attributes.

data = data.drop(data[data.payment_value > 3500].index)
data = data.reset_index(drop=True)


Correlation:
In order to visualize the correlation we can either print an heatmap matrix in order to see the correlation between each couple of variable or use a pairplot, which allows us to plot pairwise relationships between variables within a large datasets like ours, so that we can visualize everything in one figure:

Correlation = data.corr()
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot = True)

sns.pairplot(data)

Plotting the distribution of continuous data variables by using the seaborn’s distplot:

Sns.distplot(data[‘price’])



Now that we have visualized out data in different ways, we can extract from it some insights. To do It we imagine to ask ourselves some questions that we consider to be interesting about the dataset which are the following:

1)	Where do most customers come from?  

2)	What are the most frequent items bought?
Here we calculated the top 10 items bought
 
And the lowest 10 items bougth
 
It turns out that Bed and bath products are the most ordered products followed by beauty products and housewares, while the lowest are musical products, followed by flowers and children clothes.
3)	Which are the most common payment types?

index     payment_type 
0         credit_card 10319 
1         boleto 2511 
2         voucher 515 
3         debit_card 368
	The result is credit card

4)	Which are the number of orders per payment type?ù
 

5)	Which is the number of orders with number of payment installments?
 


-	RFM Analysis:
We now dive into the RFM analysis. 
As we already explained the meaning of RFM, let us show the steps that we performed to compute it keeping in mind that for each of the three components we followed the same “schema”: we started by computing the score and then we calculated the mean, standard deviation, the maximum and the minimum for each score.
1)	RECENCY:
This is the time since a customer’s last purchase and we calculated it by subtracting the customer’s last shopping date from each shopping timestamp. For those customers who made more than one purchase we kept in consideration only the most recent one.
2)	FREQUENCY:
As this score represents the total number of purchases made by a customer, it is straight forward that we just had to count the number of purchases per customer, considering each customer’s unique id.
3)	MONETARY VALUE:
Also this computation is quite fast and intuitive. We calculated it by summing all the payment values

After computing the RFM scores, we want to convert them into a single variable from 1 to 5 in order to create a first segmentation of the customers based on the RFM. Then we will assign each customer to a cluster based on its values of mean, standard deviation, max and min.



![Kmeans Clustering 3D.png](https://github.com/Albofornari/275841/blob/main/Images/Kmeans%20Clustering%203D.png)
