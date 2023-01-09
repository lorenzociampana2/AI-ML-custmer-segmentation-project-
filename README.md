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
EDA (Exploratory Data Analysis):

This process is very important for every type of data analysis project as it allows us to have an overall overview of the data, to understand if there are any type of anomalies or null values or just to make some initial assumptions of the data frame. 
In our case it was very propaedeutic because the majority of the future steps won’t have been that precise without a proper data cleaning.
Here we have some examples of our EDA with some plots:

data.head()

0,53cdb2fc8bc7dce0b6741e2150273451,b0830fb4747a6c6d20dea0b8c802d7ef,delivered,2018-07-24 20:41:37,2018-07-26 03:24:27,2018-07-26 14:31:00,2018-08-07 15:27:45,2018-08-13 00:00:00,boleto,1,...,289cdb325fb7e7f891c38608bf9e0962,2018-07-30 03:24:27,118.7,22.76,belo horizonte,SP,perfumaria,29,178,perfumery
1,86674ccaee19790309333210917b2c7d,1b338293f35549b5e480b9a3d7bbf3cd,delivered,2018-08-09 11:37:35,2018-08-09 14:35:19,2018-08-10 14:34:00,2018-08-14 18:51:47,2018-08-22 00:00:00,credit_card,5,...,289cdb325fb7e7f891c38608bf9e0962,2018-08-13 14:31:29,116.9,18.92,belo horizonte,SP,perfumaria,29,178,perfumery
2,aee682982e18eb4714ce9f97b15af5e2,8858442ea4d5dc5bb9e118e8f728095d,delivered,2018-07-09 18:46:28,2018-07-11 03:45:45,2018-07-11 15:01:00,2018-07-12 18:14:35,2018-07-18 00:00:00,boleto,1,...,289cdb325fb7e7f891c38608bf9e0962,2018-07-13 03:45:45,118.7,9.34,belo horizonte,SP,perfumaria,29,178,perfumery
3,d543201a9b42a1402ff97e65b439a48b,971bf8f42a9f8cb3ead257854905b454,delivered,2018-08-21 10:00:25,2018-08-21 10:50:54,2018-08-22 15:21:00,2018-08-28 18:58:22,2018-09-10 00:00:00,credit_card,2,...,289cdb325fb7e7f891c38608bf9e0962,2018-08-23 10:50:54,116.9,22.75,belo horizonte,SP,perfumaria,29,178,perfumery
4,d543201a9b42a1402ff97e65b439a48b,971bf8f42a9f8cb3ead257854905b454,delivered,2018-08-21 10:00:25,2018-08-21 10:50:54,2018-08-22 15:21:00,2018-08-28 18:58:22,2018-09-10 00:00:00,credit_card,2,...,289cdb325fb7e7f891c38608bf9e0962,2018-08-23 10:50:54,116.9,22.75,belo horizonte,SP,perfumaria,29,178,perfumery


We converted the variables into dates by using Pandas and then we distincted date from time:

dates = ['order_purchase_timestamp', 'order_approved_at', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date']
for i in dates:
    data[i] = pd.to_datetime(data[i])

data['order_date'] = [d.date() for d in data['order_purchase_timestamp']]
data['order_time'] = [d.time() for d in data['order_purchase_timestamp']]



data.describe() #dataset visualization with all the values

data.columns

data.nunique() #returns the output of number of unique values 

data.isnull().sum() #sum of all the null values of each variable

Removing duplicates: 

data.duplicated().sum()
data.drop_duplicates(keep='first', inplace=True)
data.reset_index(drop=True, inplace=True)

Outliers Analysis: 
analyzing some variables, we noticed that in some plots there are data points that exceed the average, for example, in ‘payment value’, which is in our mind one of the principal variable to study for a big firm, we can suppose that there are some customers who have paid an amount of money which is relatively higher than the others and this can affect our analysis in the future:

plt.figure() 
data.reset_index().plot(kind='scatter', x='index', y='payment_value', c='gray')
![payment_value_outliers.png](payment_value_outliers.png)
 
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
![customers per state.png](customers%20per%20state.png)

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

4)	Which are the number of orders per payment type?
 

5)	Which is the number of orders with number of payment installments?
 


RFM Analysis:

We now dive into the RFM analysis. 
As we already explained the meaning of RFM, let us show the steps that we performed to compute it keeping in mind that for each of the three components we followed the same “schema”: we started by computing the score and then we calculated the mean, standard deviation, the maximum and the minimum for each score.
1)	RECENCY:
This is the time since a customer’s last purchase and we calculated it by subtracting the customer’s last shopping date from each shopping timestamp. For those customers who made more than one purchase we kept in consideration only the most recent one.
2)	FREQUENCY:
As this score represents the total number of purchases made by a customer, it is straight forward that we just had to count the number of purchases per customer, considering each customer’s unique id.
3)	MONETARY VALUE:
Also this computation is quite fast and intuitive. We calculated it by summing all the payment values

After computing the RFM scores, we want to convert them into a single variable from 1 to 5 in order to create a first segmentation of the customers based on the RFM. Then we will assign each customer to a cluster based on its values of mean, standard deviation, max and min.
For what concerns Recency, we know that the mean is 73 days, so we thought that giving 4 to those customers that haven't bought for 20-65 days was the best option. For the other intervals of time, we kept it quite large considering that the strandard deviation for this score is 42 days.
With frequency we have been instead a little bit more ‘generous’ because, considering that the maximum amount of items bought is 13 and that, on average, the items bought are between 1 anmd 2, we think that a customer who bought more than 9 items can be considered a customer of level 5.
The mean for monetary value was 395, which seems to be quite a normal average of money spent, so we assigned to level 5 only who spent more than 500 euros and we thought that it was fair to go down 1 level proportiionally to 100 euros spent.
After that, as we wanted, we performed a first classification of the customers based on the previous numerical values that we assigned by recognizing 10 different types of customers, which are: hibernating, at risk, customers that can't be lost, about to sleep (almost lost), need attention, loyal customers, promising customers, new customers, potential loyalists, TOP customers.
To have a clearer visualization of the dataset after all this computations, we created new dedicated columns in order to create a new dataframe "rfm_data" where we can view only the columns of interest.
In conclusion of this first data analysis part, we plotted a pie chart to see the percentage of each group of customers and a distplot, which allows us to see the distributino of each score of RFM.


CLUSTERING ALGORITHMS:

Let's now dive into the clustering algorithm application, so that we can do a deeper investigation on our dataset and on our customers, now that we have a lot of informations about them.
Before starting with the algorithms themselves, let us explain which validation metrics we used to evaluate the efficency of each method:

- Davies Bouldin score: The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score. The minimum score is zero, with lower values indicating better clustering.
- Silhouette Score: Silhouette score or silhouette coefficient is a metric whose values are in the interval [-1, 1]. The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.
- Calinski Harabasz Score: can be used to evaluate the model when ground truth labels are not known where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset. Higher value of CH index means the clusters are dense and well separated, although there is no “acceptable” cut-off value.

After that, we applied the three methods that we chose and here we have the results 

PRINCIPAL COMPONENT ANALYSIS:

The goal of PCA is to identify the most meaningful basis to re-express data. This new basis will filter out the noise and reveal hidden structures. We want the new basis to be a linear combination of the original basis.

Let X be a dataset with m observations and n features, so X is an m × n matrix. Let Y be a new representation of X, another m × n matrix related to X by a linear transformation P (which itself is an n × n matrix). PCA will find P that transforms X to Y linearly, that is, XP=Y. The columns of P are a new set of basis vectors for representing observations (rows) in X, and are called principal components of X.

Taking as an example our dataset, it would be impossible to represent it in a scatterplot as the number of features that we take in account (and, consequently, the number of dimensions that will be required to create this scatterplot) is too big. PCA in this case is very useful in order to extract as much information as possible to create a two-dimensional representation for our data.

The clustering process follows the K-Means method, so we use WSS to create and plot the Elbow method to find the optimal number of clusters (that is 4).

Then we convert our dataset from multidimensions to 2 dimensions thanks to PCA.
pca = PCA(2)
data2 = pca.fit_transform(rfm_std)

Next we plot and check the variance of the components.

Finally we can visualize our scatterplot including our 2 Principal Components created thanks to PCA and the data divided in the 4 clusters with the corresponding centroids. The scores will be more accurate than the ones of K-Means clustering as we selected a smaller, but more significant, part of the dataset.
