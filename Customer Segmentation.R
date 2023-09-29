 # In this report, we are going to develop customer
 # segmentation to define the marketing strategy
 # for credit card company. To fulfill this
 # objective, an unsupervised learning method,
 # K-Means Clustering will be used. The process
 # inclused Data Preparation, Exploratory Data
 # Analysis, Data Pre-Processing, K-Means Clustering,
 # K-Means Clustering, Principle Component
 # Analysis(PCA), and conclusion.


install.packages('tidyverse')
install.packages('DT')
install.packages('GGally')
install.packages('RColorBrewer')
install.packages('ggplot2')
install.packages('ggforce')
install.packages('concaveman')
install.packages('factoextra')
install.packages('FactoMineR')

# Libraries Required
library(tidyverse)
library(DT)
library(GGally)
library(RColorBrewer)
library(ggplot2)
library(ggforce)
library(concaveman)
library(factoextra)
library(FactoMineR)

# Data Input
data <- read.csv("Dataset.csv")

datatable(data, options = list(scrollX = TRUE))


# Checking Data Types
str(data)

# Calculating the percentage of missing values
colSums(is.na(data)/nrow(data))

# Checking Missing Values
colSums(is.na(data))

# Over here, since the missing value in some
# columns are still below 5% of the data
# observation, the row with any missing value in
# will be dropped.
# Also, CUST_ID won't affect the clustering as it's
# unique for each observation, we are going to drop
# it.

# Drop NA
data_na <- data %>%
  drop_na(CREDIT_LIMIT, MINIMUM_PAYMENTS)

data_na

# Checking the Number of missing values after 
# dropping the rows with missing values
colSums(is.na(data_na))

# Cheking new dimensions of the dataset
dim(data_na)

# dropping CUST_ID columns
data_clean <- data_na %>%
  select(-CUST_ID)

data_clean

#checking the dimension of cleaned data
dim(data_clean)


# After data cleansing is performed, our data 
# contains 8636 observations and 17 variables. All
# data types have been converted to the desired
# data types and there's no more missing value.

# Exploratory Data Analysis
# It is the phase where we explore the data
# variables,and find out any pattern that can 
# indicate any kind of correlation between the
# variables.

# Data Summary
summary(data_clean)

# It is important to note that the dataset need to
# have the same scale. Hence, a further scaling 
# might be needed.

# Now, we are going to check the correlation of 
# these variables through data visualization.

# Correlation of each variable
ggcorr(data_clean, hjust=1, layout.exp = 2, label = T,
       label_size = 4, low = "#7d9029", mid = "white",
       high = "#3580d2")


# It can be seen that there is a strong correlation between some
# variables from the data, such as between PURCHASES and 
# ONOFF_PURCHASES, PURCHASES_FREQUENCY and ONEOFF_PURCHASES_FREQUENCY,
# etc. This result indicates that this dataset has multicolinearity
# and might not be suitable for various classification algorithms
# that have non-multicollinearity as their assumption.

# Principal Component Analysis will be performed on this data
# to produce non-multicollinearity data, while also reducing the
# dimension of the data and retaining as much as information 
# possible. The result of this analysis can be utilized further
# for classification purpose with lower computation.

# Data Pre-Processing

# Since the dataset used isn't on the same scale, we will scale
# them first using z scaling.

# Data Scaling
data_z <- scale(data_clean)
data_z

summary(data_z)

# Scaling process is done, and all of the variables have the
# same scale now.


# K-Means Clustering

# The Optimal k

# - Before the cluster analysis begin, first we must determine the
# optimal number of cluster for the analysis. To choose the
# optimal value of k in K-Means Clustering, we can use several
# methods.

# 1. Elbow Method:

fviz_nbclust(data_clean, FUNcluster = kmeans, method = "wss",
             k.max = 10, print.summary = TRUE) + labs(subtitle = "Elbow method")

# As the name suggests, in Elbow Method, we pick the elbow of the
# curve as the number of clusters to use. Based on the plot above,
# we ca see that k = 2, k = 4.

# 2. Silhouette Method

fviz_nbclust(data_clean, FUNcluster = kmeans, method = "silhouette", 
             k.max = 10, print.summary = TRUE) + labs(subtitle = "Silhouette Method")

# In Silhouete Method, the optimal number of clusters is
# chosen by the number of cluster with the highest 
# silhouette score(the peak). Based on the plot, we can see
# that k = 2 or k = 4. Hence we will try 2 types of 
# clustering with k = 2 and k = 4

# 3. Gap Statistics

fviz_nbclust(data_clean, FUNcluster = kmeans, method = "gap_stat", 
             k.max = 10, print.summary = TRUE) + labs(subtitle = "Gap Statistics")


# Clustering

# K-Means with k = 2

set.seed(123)
data_KM2 <- kmeans(x = data_z, centers = 2)

# Number of observations in each cluster
data_KM2$size

# Location of the center of the cluster/centroid, commonly
# used for cluster profiling
data_KM2$centers

# After the clustering process, it can be seen that we have 5015 observations
# in cluster 1 and 3621 observations in cluster 2. We can check the 
# visualization of it using PCA bi plot as below.

# Cluster Visualization
fviz_cluster(object = data_KM2, data = data_z, geom = "point") + 
  ggtitle("K-Means Clustering Plot") +
  scale_color_brewer(palette = "Accent") + theme_minimal() + 
  theme(legend.position = "bottom")


# K-Means with k = 4

set.seed(123)
data_KM4 <- kmeans(x = data_z, centers = 4)

# Number of observations in each cluster
data_KM4$size

# Location of the center of the cluster/centroid, 
# commonly used for cluster profiling.

data_KM4$centers

# After the clustering process, it can be seen
# above that we have 3984 observations in cluster 1,
# 2934 observations in cluster 2, 1068 observations
# in cluster 3, and 650 observations in cluster 4.
# We can check the visualization of it using PCA
# biplot as below:

# Cluster Visualization:
fviz_cluster(object = data_KM4, data = data_z,
            geom = "point") + 
  ggtitle("K-Means Clustering Plot") +
  scale_color_brewer(palette = "Accent") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Since in the plot the dimension was reduced into two-dimensional, each of the
# cluster intersect each other, some even clumped together. This happens because
# we have little dimensions to represent our data (17 variables).

# Goodness of Fit

# The evaluation of clustering can be seen from 3 values:

# Within Sum of Squares: the sum of squared distances from each observation to
# the centroid of each cluster.

# Between Sum of Squaress: the sum of squared distances from each centroid of each
# cluster to the whole data average.

# Total Sum of Squares: the sum of squared distances from each observation to
# the whole data average.

# Goodness of Fit of K-Means with k = 2.

# Within Sum of Squares
data_KM2$withinss

# Total Sum of Squares
data_KM2$betweenss/data_KM2$totss

# The "good" cluster results have a low value of Within Sum of Squares and 
# Total Sum of Squares near to 1. However, as can be seen from the result, the
# Within Sum of Squares are too big and the Total Sum of Squares are still far
# from 1. Which indicates that this clustering might not be good enough.

# Goodness of Fot of K-Means with k = 4.

# Within Sum of Squares
data_KM4$withinss

# Total Sum of Sqaures
data_KM4$betweenss/data_KM4$totss

# The same with the previous section, “good” cluster results have a
# low value of Within Sum of Squares and Total Sum of Squares near to
# 1. Unfortunately, as can be seen from the result, the Within Sum of
# Squares are too big and the Total Sum of Squares are still far from 1.
# Which indicates that this clustering might not be good enough. However,
# since this result is better than k = 2, we are going to use k = 4 model
# for further analysis.

# Cluster Profiling

# Now that we get the information about the cluster of each observation,
# we are going to combine the cluster column into the data set to interpret
# each characteristics of the cluster. We could also add the CUST_ID column
# to find out which cluster each customer belongs to in the end.

# Combining the cluster label into the data set
data_clean$CLUSTER <- as.factor(data_KM4$cluster)

# Profiling with aggregation table
data_clean %>%
  group_by(CLUSTER) %>%
  summarise_all(mean)


# Profiling with aggregation table

data_clean %>%
  group_by(CLUSTER) %>%
  summarise_all(mean) %>%
  tidyr::pivot_longer(-CLUSTER) %>%
  group_by(name) %>%
  summarize(cluster_min_val = which.min(value),
            cluster_max_val = which.max(value))


ggplot(data_clean, aes(x = factor(CLUSTER), y = PURCHASES, fill = CLUSTER, colour = CLUSTER)) + 
  geom_bar(stat = "identity", position = "dodge")

# As can be seen from the plot, cluster 1 & 4 have the lowest amount of
# purchases compared with the other clusters, hence if there are offers 
# such as reward programs, discounts using credit card, they could be the
# best target.

ggplot(data_clean, aes(x = factor(CLUSTER), y = PAYMENTS, fill = CLUSTER, colour = CLUSTER)) + 
  geom_bar(stat = "identity", position = "dodge")

# As can be seen from the plot, cluster 2 & 3 also has the highest amount
# of payments compared with the other clusters, indicating how aware they are
# of their credits. Hence, if there are offers such as loyalty points, they
# could be the best target. From the both plots above, cluster 4 relatively
# has the lowest amount of purchases and payments compared with the other
# clusters. They also can be offered to zero interest program to increase
# theirs purchase and payments.

# Now the question is which cluster do they belong to?

# We could add the CUST_ID column back to find out which cluster each
# customer belongs to.

# Combining the 'CUST_ID' column into the data set
data_ID <- data_clean %>%
  mutate(CUST_ID = data_na$CUST_ID)

datatable(data_ID, options = list(scrollx = TRUE))
# To find out a certain customer belongs to which cluster, please type
# in their ID in the Search tab from the data table above.

# Principal Component Analysis(PCA)

# Dimensionality Reduction

# Dimensionality reduction are often performed using PCA as the algorithm
# , the main purpose is to reduce the number of variables (dimensions) in
# the data while retaining as much information as possible. Dimensionality
# reduction can solve the problem of high-dimensional data such as in the
# data set we are using.

# PCA using FactoMineR
data_pca <- PCA(X = data_clean, quali.sup = 18, 
                scale.unit = T, ncp = 17, graph = F)
data_pca$eig

# Variance explained by each dimensions
fviz_eig(data_pca, ncp = 17, addlabels = T, 
         main = "Variance explained by each dimensions")

# Based from the above results, if we want to retain 80% information
# of the data set, using 7 dimensions to do so is enough. This means
# that we can reduce the number of dimensions on our data set from 17
# to 7 dimensions. However, by doing so,it might be hard to interpret
# the classification result, since we can interpret it by each variable
# anymore. But, we could see the Variable Contribution of each PCA, such
# as below.

# Variable Contribution of PC1
fviz_contrib(X = data_pca, choice = "var", axes = 1)


# Variable contribution untuk PC2
fviz_contrib(X = data_pca, choice = "var", axes = 2)

# PCA Visualization

# Individual Factor Map
# - Individual Factor Map plot the distribution of observations
# to find out which index is considered an outlier.

plot.PCA(x = data_pca, choix = "ind", invisible = "quali", 
         select = "contrib 8", habillage = "CLUSTER") + 
  scale_color_brewer(palette = "Accent") + 
  theme(legend.position = "bottom")

# From the plot, we can see 8 outliers' indexes: 123, 247,
# 465, 513, 1167, 1510, 3793, and 2055. Most of the outliers
# are in CLUSTER_2.

# Variables Factor Map

# Variables Factor Map are used to find out the variable
# contribution to each PC, as well as the amount of information
# summazrized from each variable to each PC; and to find out the
# correlation between the initial variables.

fviz_pca_var(data_pca, select.var = list(contrib = 17), col.var = "contrib", 
             gradient.cols = c("red", "white", "blue"), repel = TRUE)

# From the plot above, we can conclude that:
#   
#   The most contributing variables in PC1 are PURCHASES and PURCHASES_TRX.
# 
# The most contributing variables in PC2 are CASH_ADVANCE and CASH_ADVANCE_TRX
# 
# Positive highly correlated variables are:
#   
#   PURCHASES and ONEOFF_PURCHASES
# PURCHASES_FREQUENCY and PURCHASES_INSTALLMENTS_FREQUENCY
# CASH_ADVANCE and CASH_ADVANCE_TRX


# Cluster Visualization with PCA

# PCA can also be integrated with the result of K-Means Clustering to
# help visualize our data in a fewer dimensions than the original features.

# visualisasi PCA + hasil kmeans clustering
fviz_pca_biplot(data_pca, habillage = 18, addEllipses = T,
                geom.ind = "point") + 
  theme_minimal() + 
  theme(legend.position = "bottom") + 
  scale_color_brewer(palette = "Accent")

# However, same problem as the previous section (clustering plot using PCA biplot)
# happens: each of the cluster intersect each other, some even clumped together. This
# happens because we have to little dimensions to represent our data (17 variables).


# Based on the data used in this report and the K_Means Clustering process that
# has been done, we can conclude that:
#   
# Cluster 1: Customers with lowest amount of all purchases, not much withdrawals,
# indicates not many transactions of the credit card compared to the other
# clusters.
# 
# Cluster 2: Customers with lowest amount of withdrawal and frequency, however,
# have the highest amount of all purchases. They have the longest tenure and
# highest percent of full payments paid, indicating that they are aware of
# their credits.
# 
# Cluster 3: Customers with high amount of balance, high cash advance and high
# credit limit. Their balance also seemed to be updated frequently, indicates
# many transactions of the credit card. The customers of this cluster also have
# high amount of minimum payments, however, lowest percent of full payments paid,
# indicating higher loans amount and often like to withdraw a lot of money
# from the credit card.
# 
# Cluster 4: Customers with lowest amount of balance and lowest credit limit.
# The customers of this cluster also have the lowest of minimum payments,
# payments and tenure; indicating that the transactions made in these credit
# cards are small transactions.
# 
# Based on the clusters that have been produced, a few business suggestions
# can be made to profit the industry, such as:
#   
# Cluster 1 & 4 have the lowest amount of purchases compared with the other
# clusters, hence if there are offers such as reward programs, discounts
# using credit card, they could be the best target.
# 
# Cluster 2 & 3 also has the highest amount of payments compared with the
# other clusters, indicating how aware they are of their credits. Hence,
# if there are offers such as loyalty points, they could be the best target.
# 
# Cluster 4 relatively has the lowest amount of purchases and payments compared
# with the other clusters. They also can be offered to zero interest program to
# increase theirs purchase and payments.
# 
# To find out a certain customer belongs to which cluster, type in their ID in
# the Search tab from the data table in this section.
# 
# 
