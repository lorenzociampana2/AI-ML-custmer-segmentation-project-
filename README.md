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
-	K-means clustering: is one of the most  used clustering method, especially in this cases. We had the opportunity in fact to look at some similar business projects on the web where this algorithm is always used, especially when we have to scale large datasets;
-	Hierarchical clustering: Hierarchical clustering is another popular method of grouping data. It creates groups so that data within a group are similar to each other and different from objects in other groups. Clusters are visually represented in a hierarchical tree called a dendrogram.
-	PCA (Principal Component Analysis): this method is very useful whenever we have to deal with large datasets as it helps us reducing the dimensionality of such datasets and minimizing the amount of useful information so that we can have a clearer and more precise interpretation of it.


PROJECT CORE:

Let’s now dive into the steps that we followed to have a clear visualization of the dataset, which can be divided as follows:
EDA (Exploratory Data Analysis):

This process is very important for every type of data analysis project as it allows us to have an overall overview of the data, to understand if there are any type of anomalies or null values or just to make some initial assumptions of the data frame. 
In our case it was very propaedeutic because the majority of the future steps won’t have been that precise without a proper data cleaning.
Here we have some examples of our EDA with some plots:

data.head()

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
analyzing some variables, we noticed that in some plots there are data points that exceed the average, for example, in ‘payment value’, which is in our mind one of the principal variable to study for a big firm, we can suppose that there are some customers who have paid an amount of money which is relatively higher than the others and this can affect our analysis in the future, as we can identify them as 'exceptions'.
You can clearly see it with the following graph:

plt.figure() 
data.reset_index().plot(kind='scatter', x='index', y='payment_value', c='gray')
![Payment value and index.png](https://github.com/Albofornari/275841/blob/main/Images/Payment%20value%20and%20index.png)
 
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
![Heatmap.png](https://github.com/Albofornari/275841/blob/main/Images/Heatmap.png)
In this type of plot, the strength of a relationship is indeed given by positive or negative values. As an exmaple of negative correlation we can take the couple 'payment_installments' and 'product_name_length' which are obviously two different characteristics of a product, which cannot be in relation.
On the other hand, the highest correlation, which is also highlighted, is the one between 'price' and 'payment_value', because they tell us very similar informations.

Then we used the seaborn's pairplot in order to have pairwise relationships between variables within a large dataset like ours, so that we can visualize everything in only one figure:

sns.pairplot(data)
![Pairplot.png](https://github.com/Albofornari/275841/blob/main/Images/Pairplot.png)

Then we plotted the distribution of continuous data variables by using the seaborn’s distplot:

Sns.distplot(data[‘price’])


Now that we have visualized out data in different ways, we can extract from it some insights. To do It we imagine to ask ourselves some questions that we consider to be interesting about the dataset which are the following:

1)	Where do most customers come from?  
![Num of costumer per state and cities with most costumers.png](https://github.com/Albofornari/275841/blob/main/Images/Num%20of%20costumer%20per%20state%20and%20cities%20with%20most%20costumers.png)

2)	What are the most frequent items bought?
Here we calculated the top 10 items bought
![Top 10 product categories.png](https://github.com/Albofornari/275841/blob/main/Images/Top%2010%20product%20categories.png)
 
And the lowest 10 items bougth
![lowest 10 products ordered.png](https://github.com/Albofornari/275841/blob/main/Images/lowest%2010%20products%20ordered.png)
It turns out that Bed and bath products are the most ordered products followed by beauty products and housewares, while the lowest are musical products, followed by flowers and children clothes.
3)	Which are the most common payment types?

index     payment_type 
0         credit_card 10319 
1         boleto 2511 
2         voucher 515 
3         debit_card 368
	The result is credit card

4)	Which are the number of orders per payment type?
![Order by payment Type.png](https://github.com/Albofornari/275841/blob/main/Images/Order%20by%20payment%20Type.png)

5)	Which is the number of orders with number of payment installments?
![Count of orders with number of payment installments.png](https://github.com/Albofornari/275841/blob/main/Images/Count%20of%20orders%20with%20number%20of%20payment%20installments.png)


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
In the end we used a distplot to see the distribution of each score of RFM.

CLUSTERING ALGORITHMS: 

Let's now dive into the clustering algorithm application, so that we can do a deeper investigation on our dataset and on our customers, now that we have a lot of informations about them.
Before starting with the algorithms themselves, let us explain which validation metrics we used to evaluate the efficency of each method:

- Davies Bouldin score: The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score. The minimum score is zero, with lower values indicating better clustering.
- Silhouette Score: Silhouette score or silhouette coefficient is a metric whose values are in the interval [-1, 1]. The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.
- Calinski Harabasz Score: can be used to evaluate the model when ground truth labels are not known where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset. Higher value of CH index means the clusters are dense and well separated, although there is no “acceptable” cut-off value.

After that, we applied the three methods that we chose and here we have the results and plots for each of them:

K-Means:
Before explaining the implementation, let us introduce you to this method, which can be resumed in 4 points:
1) The algorithm randomly chooses a centroid (which is the center of a cluster) k.
2) It assigns every data point in the dataset to the nearest centroid, meaning that if that data point is closer to that centroid, it belongs to its cluster
3) For every cluster, the algorithm recomputes the centroid by taking the average of all the point in the cluster. Since the centroids change, then it assigns the data points to new clusters.
4) The same process is repeated until no changes in centroids values are produced.
![Elbow method km.png](https://github.com/Albofornari/275841/blob/main/Images/Elbow%20method.png)
The first step for the implementation of K-means algorithm was to apply the elbow method in order to find the optimal number of clusters. In our case we had to choose between 3, 4 or 5 clusters. We decided to consider 4 because they seemed to have the clearest scores
![Kmeans Clustering 3D.png](https://github.com/Albofornari/275841/blob/main/Images/Kmeans%20Clustering%203D.png)
From the 3D representation of the Kmeans algorithm we can clearly see the 4 clusters. The 3D representation allows us to have a better understanding of the clustering and a clear graphical representation compared to the two-dimensional graph. Form the graph above we can see that Cluster 1(orange cluster) contain costumers that have a spent a low amount of money and with a very low frequency however the last purchase resulted to be recent. Cluster 2(yellow cluster) seems to contains active customers that have a good recency and also an relatively high frequency. Cluster 3(brown cluster) we can deduce that it represents customers that are no longer spending in the store because based on the recency the last purchase was done a long time ago. Cluster 4(purple cluster) represents the customers that are now spending the most in the stores with an high frequency and a high monetary value

Silhouette score: 0.497172; Calinski Harabasz score: 10853.192306; Davies Bouldin: 0.745792


Hierarchical Clustering: 
In this method, clusters are built by creating a tree-based hierarchy which is often represented by a dendogram, which you will see in the nex lines.
Unlike other methods such as K-means, the number of clusters is decided by the user and the clustering proces is said to be 'deterministic', since assignments won't change as the algorithm is executed multiple times on the same input data.

![Dendogram HC.png](https://github.com/Albofornari/275841/blob/main/Images/Dendogram%20HC.png)
For what concerns Hierarchical clustering method, we found the optimal number of clusters by using the dendogram, which uses branches of clusters to show how closely objects are related to one another, so those clusters that are located on the same height level are more closely related than clusters that are located on different height levels.
Also in this case the optimal number was clearly 4.
![Hierarchical Clustering 3D.png](https://github.com/Albofornari/275841/blob/main/Images/Hierarchical%20Clustering%203D.png)
Here, we also have a 3D representation of the clusters.

Silhouette Score: 0.457153; Calinski Harabasz Score: 8814.023578; Davies Bouldin Score: 0.825724

PCA: 
The goal of PCA is to identify the most meaningful basis to re-express data. This new basis will filter out the noise and reveal hidden structures. We want the new basis to be a linear combination of the original basis.
Let X be a dataset with m observations and n features, so X is an m × n matrix. Let Y be a new representation of X, another m × n matrix related to X by a linear transformation P (which itself is an n × n matrix). PCA will find P that transforms X to Y linearly, that is, XP=Y. The columns of P are a new set of basis vectors for representing observations (rows) in X, and are called principal components of X.
Taking as an example our dataset, it would be impossible to represent it in a scatterplot as the number of features that we take in account (and, consequently, the number of dimensions that will be required to create this scatterplot) is too big. PCA in this case is very useful in order to extract as much information as possible to create a two-dimensional representation for our data.
The clustering process follows the K-Means method, so we use WSS to create and plot the Elbow method to find the optimal number of clusters (that is 4).
Then we convert our dataset from multidimensions to 2 dimensions thanks to PCA. 

Silhouette score: 0.497176; Calinski Harabasz score. 10853.142409; Davies Bouldin score: 0.745792

As you can see the difference between the majority of the scores is very small, but if we had to choose a 'winner' among the algorithms we can say that K-means, as we also expacted, has been the most efficient method. It has in fact the hoghest score in the Silhouette and Calinski scores, which, as we already explained, need to be as high as possible to tell us that the algorithm performed well.



PRINCIPAL COMPONENT ANALYSIS: 

The goal of PCA is to identify the most meaningful basis to re-express data. This new basis will filter out the noise and reveal hidden structures. We want the new basis to be a linear combination of the original basis.
Let X be a dataset with m observations and n features, so X is an m × n matrix. Let Y be a new representation of X, another m × n matrix related to X by a linear transformation P (which itself is an n × n matrix). PCA will find P that transforms X to Y linearly, that is, XP=Y. The columns of P are a new set of basis vectors for representing observations (rows) in X, and are called principal components of X.
Taking as an example our dataset, it would be impossible to represent it in a scatterplot as the number of features that we take in account (and, consequently, the number of dimensions that will be required to create this scatterplot) is too big. PCA in this case is very useful in order to extract as much information as possible to create a two-dimensional representation for our data.
The clustering process follows the K-Means method, so we use WSS to create and plot the Elbow method to find the optimal number of clusters (that is 4).
Then we convert our dataset from multidimensions to 2 dimensions thanks to PCA.  

![Pca barplot.png](https://github.com/Albofornari/275841/blob/main/Images/Pca%20barplot.png)



pca = PCA(2)
data2 = pca.fit_transform(rfm_std)

Next we plot and check the variance of the components.
Finally we can visualize our scatterplot including our 2 Principal Components created thanks to PCA and the data divided in the 4 clusters with the corresponding centroids. The scores will be more accurate than the ones of K-Means clustering as we selected a smaller, but more significant, part of the dataset.


CONCLUSIONS

All the work above was done to find some interesting values such as the minimim, maximum, mean and median of the K means and the Hierarchical clustering technique. However after all the analysis done and the graphical representations of the clustering methods, we can deduce that our Brazilian customers can clrearly be divided into four different groups. This four groups all have different characteristics; especially we have a group costumers that are now spending in our stores and costumers that are no longer doing it and we can notice this thanks to the Recency. We also have a group of costumers that are actively spending a high amount of money in our stores and this can be highlighted in every clustering procedure.
In conclusion, we would definitely invest in advertising addressed to those customers that are not longer doing shopping in our stores, for example giving away some little discounts or coupons, while trying to keep the good relationship with those ones who are used to spend a higher amount of money.