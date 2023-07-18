---
cssclass: greenTek, wideTable
---
## INDEX

| ***No.*** | ***Question***                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| 1.        | What is machine learning, and how is it different from traditional programming?                               |
| 2.        | Explain the difference between supervised and unsupervised learning.                                          |
| 3.        | What is the difference between regression and classification in machine learning?                             |
| 4.        | Describe the k-nearest neighbor (KNN) algorithm and its applications.                                         |
| 5.        | What is overfitting in machine learning, and how can it be prevented?                                         |
| 6.        | Explain the concept of bias and variance in machine learning.                                                 |
| 7.        | What is cross-validation, and why is it important in machine learning?                                        |
| 8.        | Describe the decision tree algorithm and its applications.                                                    |
| 9.        | Explain the concept of feature selection in machine learning.                                                 |
| 10.       | Describe the logistic regression algorithm and its applications.                                              |
| 11.       | Explain the concept of support vector machines (SVM) in machine learning.                                     |
| 12.       | Describe the k-means clustering algorithm and its applications.                                               |
| 13.       | What is the difference between clustering and classification in machine learning?                             |
| 14.       | Explain the concept of gradient descent in machine learning.                                                  |
| 15.       | Describe the Naive Bayes algorithm and its applications.                                                      |
| 16.       | What is ensemble learning, and how is it used in machine learning?                                            |
| 17.       | What is the difference between batch and online learning?                                                     |
| 18.       | What is the difference between supervised and unsupervised reinforcement learning?                            |
| 19.       | Explain the bagging technique and its applications.                                                           |
| 20.       | Describe the principal component analysis (PCA) algorithm and its applications.                               |
| 21.       | Explain the difference between a decision tree and a random forest algorithm.                                 |
| 22.       | What is Recursive Feature Elimination (RFE)? How can it be used in Machine Learning?                          |
| 23.       | What is Variance Inflation Factor (VIF)? How can it be used to address multicollinearity in Machine Learning? |
| 24.       | What is the ROC curve? How is it used in evaluating classification models?                                    |
| 25.       | What is precision and recall in Machine Learning? How are they used in evaluating classification models?      |
| 26.       | Explain the difference between parametric and non-parametric Machine Learning algorithms.                     |
| 27.       | What is the curse of dimensionality in Machine Learning? How can it be addressed?                             |
| 28.       | What is an outlier in Machine Learning? How can it be detected and handled?                                   |
| 29.       | Explain the concept of feature scaling in Machine Learning.                                                   |
| 30.       | What is the difference between a validation set and a test set in Machine Learning?                           |
| 31.       | What are the steps involved in a typical Machine Learning project?                                            |
| 32.       | What is the difference between a linear and nonlinear regression in Machine Learning?                         |
| 33.       | What is the purpose of dimensionality reduction in Machine Learning?                                          |
| 34.       | What is the difference between overfitting and underfitting in Machine Learning?                              |
| 35.       | What is the purpose of hyperparameter tuning in Machine Learning?                                             |
| 36.       | Explain the difference between a simple and a multiple linear regression in Machine Learning.                 |
| 37.       | Explain the concept of feature engineering in Machine Learning.                                               |
| 38.          |  Explain the concept of the confusion matrix in Machine Learning.                                                                                                             |

***
<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

#### 1. What is machine learning, and how is it different from traditional programming?
***
**Ans :**  
Machine Learning is a field that consists of learning algorithms that:
1. Improve their performance *P*
2.  At executing some task *T*
3.  Over time with experience *E*

Machine learning is a subset of artificial intelligence (AI) that focuses on developing algorithms and models that allow computers to learn and make predictions or decisions without being explicitly programmed. It involves creating mathematical models and training them on large amounts of data to identify patterns, extract meaningful insights, and make accurate predictions or decisions.

The key difference between machine learning and traditional programming lies in how the computer system acquires knowledge or information. In traditional programming, a programmer writes explicit rules and instructions for the computer to follow in order to solve a specific problem. The program executes those instructions, and the desired output is obtained.

On the other hand, in machine learning, the computer system learns from data rather than relying solely on explicit instructions. Instead of programming specific rules, the machine learning algorithm is trained on a labeled dataset or through reinforcement, allowing it to automatically learn patterns and relationships within the data. The algorithm then uses this learned knowledge to make predictions or decisions on new, unseen data.

Machine learning offers several advantages over traditional programming, including:

1. **Handling complex problems:** Machine learning can effectively handle complex tasks with large amounts of data and intricate patterns that may be challenging to program explicitly.
2. **Adaptability:** Machine learning models can learn and adapt to new data, making them suitable for dynamic and changing environments.
3. **Automation:** Once trained, machine learning models can automate decision-making processes without human intervention.
4. **Discovering insights:** Machine learning algorithms can discover hidden patterns, correlations, and insights in data that may not be readily apparent to humans.
5. **Scalability:** Machine learning allows for scalability, as models can process large datasets and make predictions or decisions in real-time.

***
#### 2. Explain the difference between supervised and unsupervised learning.
***
**Ans :**

| ***Supervised Learning***                                                                                                                                              | ***Unsupervised Learning***                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Supervised learning algorithms are trained using labeled data.                                                                                                         | Unsupervised learning algorithms are trained using unlabeled data.                                                |
| Supervised learning model takes direct feedback to check if it is predicting correct output or not.                                                                    | Unsupervised learning model does not take any feedback.                                                           |
| Supervised learning model predicts the output.                                                                                                                         | Unsupervised learning model finds the hidden patterns in data.                                                    |
| In supervised learning, input data is provided to the model along with the output.                                                                                     | In unsupervised learning, only input data is provided to the model.                                               |
| The goal of supervised learning is to train the model so that it can predict the output when it is given new data.                                                     | The goal of unsupervised learning is to find the hidden patterns and useful insights from the unknown dataset.    |
| Supervised learning can be categorized in Classification and Regression problems.                                                                                      | Unsupervised Learning can be classified in Clustering and Associations problems.                                  |
| Supervised learning can be used for those cases where we know the input as well as corresponding outputs.                                                              | Unsupervised learning can be used for those cases where we have only input data and no corresponding output data. |
| It includes various algorithms such as Linear Regression, Logistic Regression, Support Vector Machine, Multi-class Classification, Decision tree, Bayesian Logic, etc. | It includes various algorithms such as Clustering, Hierarchal Clustering, and Apriori algorithm.                  |

***
#### 3. What is the difference between regression and classification in machine learning?
***
**Ans :**

| ***Regression***                                                                                                           | ***Classification***                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Regression is a process of finding the correlations between dependent and independent variables.                           | Classification is a process of finding a function which helps in dividing the dataset into classes based on different parameters.                                      |
| In Regression, the output variable must be of continuous nature or real value.                                             | In Classification, the output variable must be a discrete value.                                                                                                       |
| The task of the regression algorithm is to map the input variable(s) with the continuous output variable(y).               | The task of the classification algorithm is to map the input variables(s) with the discrete output variable(y).                                                        |
| Regression Algorithms are used with continuous data.                                                                       | Classification Algorithms are used with discrete data.                                                                                                                 |
| In Regression, we try to find the best fit line, which can predict the output more accurately.                             | In Classification, we try to find the decision boundary, which can divide the dataset into different classes.                                                          |
| Regression algorithms can be used to solve the regression problems such as Height Prediction, House price prediction, etc. | Classification Algorithms can be used to solve classification problems such as Identification of spam emails, Speech Recognition, Identification of cancer cells, etc. |
| The regression Algorithm can be further divided into Linear and Non-linear Regression.                                     | The Classification algorithms can be divided into Binary Classifier and Multi-class Classifier.                                                                        |

***
#### 4. Describe the k-nearest neighbor (KNN) algorithm and its applications.
***
**Ans :**
k-Nearest Neighbor (KNN) algorithm is widely used for classification problems in machine learning. It is one of the simplest supervised machine learning algorithms. It is extremely easy to implement in its most basic form, and yet performs quite complex classification tasks. KNN is a lazy learning algorithm since it doesn't have a specialized training phase. Rather, it uses all of the data for training while classifying a new data point or instance.

The algorithm simply calculates the distance of a new data point to all other training data points. The distance can be of any type e.g. Euclidean or Manhattan etc. It then selects the K-nearest data points, where *K* can be any integer specified by the user. Finally, it assigns the data point to the class to which the majority of the *K* data points belong.

Suppose, we have a dataset with two variables, which when plotted looks like Figure 1. 

![[knn_figure1 | 400]]

Our task is to classify a new data point *Z* into *Brown* class or *Green* class. The coordinate values of the data point are *x = 34* and *y = 25*. Suppose the value of *K* is *3*. The KNN algorithm starts by calculating the distance of point *Z* from all the points. It then finds the 3 nearest points with least distance to point *Z*. This is show in figure below. The 3 nearest points have been encircled as shown in the below figure.

![[knn_figure2 | 400]]

The final step of the KNN algorithm is to assign new point to the class to which majority of the three nearest points belong. From the figure above, we can see that the two of the nearest points belong to the class *Green* while one belongs to the class *Brown*. Therefore, the new data point will be classified as *Green*.

Implementing the above algorithm is pretty easy and can be done in python by importing the *KNeighborsClassifier* from the *sklearn.neighbors* library.

***Applications of the algorithm :***
- It’s used in many different areas, such as handwriting detection, image recognition, and video recognition. KNN is most useful when labeled data is too expensive or impossible to obtain, and it can achieve high accuracy in a wide variety of prediction-type problems.
- **Computer Vision**: KNN performs classification tasks. It handles image data well, and it’s considered a fine option for classifying a bunch of diverse images based on similarities.
- **Content Recommendation**: KNN is great for content recommendation. It’s used in many recommendation system engines and continues to be relevant even though there are newer, more powerful systems already available.

***
#### 5. What is overfitting in machine learning, and how can it be prevented? --- OR --- What is Overfitting in Machine Learning? How can it be addressed?
***
**Ans :**
Overfitting is a concept in data science, which occurs when a statistical model fits exactly against its training data. When a model performs excellent on training data, but has rather poor performance on test data, it is said that the model has overfitted the training data. An overfit model can give inaccurate predictions and cannot perform well for all types of new data. It can be identified with numerous ways, one being the case where there is a difference of 5% or more between Training and Testing accuracy.

***Possible Reasons for Overfitting:***
1. **High variance and low bias:** Overfitting often occurs when a model has high variance and low bias. High variance means that the model is sensitive to the noise or fluctuations in the training data. Low bias refers to the model's inability to capture the underlying patterns or relationships in the data.
2. **The model is too complex:** If a model is overly complex, it has a higher risk of overfitting. Complex models, such as those with a large number of parameters or high-degree polynomial functions, sometimes cause the model to learn noise or irrelevant patterns, leading to poor generalization.
3. **The size of the training data:** It can be either that training data size is too small and does not contain enough data samples to accurately represent all possible input data values or that the training data contains large amounts of irrelevant information, called noisy data.

We can prevent model overfitting using many different techniques:
1. **Introduce a validation dataset:**  Validation dataset is used to provide an unbiased evaluation after training the model on the training dataset. It’s helpful during the design iteration of the architecture, or hyperparameter tuning of the model.
2. **Cross-validation:** Cross-validation splits the data such that the validation data is representative of both training and the data from the real-world scenario. This helps the model generalize well and yield good results.
3. **Hyper-parameter Tuning:** Hyperparameter tuning is the process of determining the right combination of hyperparameters that maximizes the model performance. We can use methods like *GridSearchCV*, *RandomizedSearchCV* to improve model performance.
4. **Ensemble Methods:** Ensemble methods combine a group of predictive models to get an average prediction. It’s a very successful method in reducing overfitting.

***
#### 6. Explain the concept of bias and variance in machine learning.
***
**Ans :**
Variance and Bias fall under the category of reducible errors in machine learning.
Reducible errors are those which can be further improved upon to improve the performance of a model.

**Bias** : Bias is the difference between the Predicted Value and the Expected Value. Bias is the simple assumptions that our model makes about our data to be able to predict new data. To explain further, the model makes certain assumptions when it trains on the data provided. When it is introduced to the testing/validation data, these assumptions may not always be correct.

Let our input variable be $x$ and target variable be $y$.  We map the relationship between the two using a function $f$. Therefore, $y = f(x) + e$. Here $e$ is the error that is normally distributed. 

The Bias of the model is: $$Bias = E[f'(x)] – f(x)$$
$E[f'(x)]$ stands for expected prediction value given input $x$. 
$f(x)$ is the true value given input $x$. 

**Variance** : The Variance is when the model takes into account the fluctuations in the data i.e. the noise as well. This value specifies the amount of variation in the prediction if a different training data was used. In simple words, variance tells that how much a random variable is different from its expected value.

Mathematically, the variance error in the model is: 
$$Variance=E[E(f'(x))-f'(x)]^2$$

A ideal performing model has low bias and low variance. However, it is not possible practically. We can however try to achieve a condition as close as possible.

![[var_and_bias | 400]]

To achieve a good performing model, it is required to make a balance between bias and variance errors, and this balance between the bias error and variance error is known as the Bias-Variance trade-off.

![[bias_var_tradeoff | 400]]
***
#### 7. What is cross-validation, and why is it important in machine learning?
***
**Ans :**
Cross-validation is a resampling method that uses different portions of the data to test and train a model on different iterations. It is a technique for validating the model efficiency by training it on the subset of input data and testing on previously unseen subset of the input data. We can also say that it is a technique to check how a statistical model generalizes to an independent dataset. 

Cross-validation is important in machine learning because it helps overcome overfitting of model on training data. If we use a model that is overfitted on training data, we get good accuracy on training data, but the model performs poorly on new data. In such cases, cross validation can help train the model more accurately with good results. Also cross validation helps achieve good results in case where overall data is limited. It also helps in comparing the results of different predictive models at once.

There are a number of cross-validation methods for datasets as follows:
1. ***K-fold :*** K-fold divides all the samples in *K* groups of samples, called folds of equal sizes (if possible). The prediction function is learned using *K−1* folds, and the fold left out is used for test.
2. ***Leave One Out :***  (or LOO) is a simple cross-validation. Each learning set is created by taking all the samples except one, the test set being the sample left out. Thus, for $n$ samples, we have $n$ different training sets and $n$ different tests set. This cross-validation procedure does not waste much data as only one sample is removed from the training set.

We can also use cross validation for hyperparameter tuning using *GridSearchCV* and *RandomizedSearchCV* methods:
1. ***GridSearchCV :*** This methods trains the model on the dataset with all possible combinations of the specified hyperparameters. It is a good choice when we have a small dataset. GridSearchCV becomes very computationally intensive as number of parameters and size of dataset increases.
2. ***RandomizedSearchCV :***  RandomizedSearchCV solves the drawbacks of GridSearchCV, as it goes through only a fixed number of hyperparameter settings. It selects the hyperparameters in a random fashion to find the best set of hyperparameters. This approach reduces unnecessary computation and provides results in less time.

***
#### 8. Describe the decision tree algorithm and its applications.
***
**Ans :**
Decision Tree is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. In a Decision tree, there are two nodes, which are *Decision Node* and *Leaf Node*. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
Decision trees can be used for Classification as well as Regression, but are mostly used for Classification problems. 

Terminologies in Decision Tree Algorithm:
- **Root Node:** Root node is from where the decision tree starts. It represents the entire dataset, which further gets divided into two or more homogeneous sets.
- **Leaf Node:** Leaf nodes are the final output node, and the tree cannot be segregated further after getting a leaf node.
- **Branch/Sub Tree:** A tree(s) formed by splitting from the tree in the previous level.

***Steps to create a Decision Tree:***
- **Step-1:** Begin the tree with the root node, says $S$, which represents the complete dataset.
- **Step-2:** Find the best attribute in the dataset using *Attribute Selection Measures (ASM)*.
- **Step-3:** Divide $S$ into subsets that contains possible values for the best attribute.
- **Step-4:** Generate the decision tree node, labelled with the best attribute.
- **Step-5:** Recursively make new decision trees using the subsets of the dataset created in *Step-3*. Continue this process until a stage is reached where you cannot further classify the nodes. The nodes in these level are also called as leaf nodes.

Example of a decision tree to classify people as fit or unfit.

![[decision_tree | 400]]

***Attributes Selection Measures (ASM) :***
**1. Information Gain (IG):** Information gain is the measurement of changes in entropy after the segmentation of a dataset based on an attribute.

$$ IG = Entropy{(S)}-\sum{[(Weighted\;Avg)} * {Entropy} {(Each\;Feature)]} $$

**Entropy:** Entropy is a metric to measure the impurity in a given attribute. It specifies randomness in data. In the above example, entropy can be calculated as:
$$Entropy_{(S)}= -P_{(fit)}\:log_2 P_{(fit)}- P_{(unfit)}\:log_2 P_{(unfit)} $$
where $P$ is the probability.

**2. Gini Index (GI):** Gini index is a measure of impurity or purity used while creating a decision tree. An attribute with the low Gini index should be preferred as compared to the high Gini index.
$$ Gini\;Index= 1- ∑\limits_{j=1}^nP_j^2 $$
$$ Here,\;P_j\;is\;the\;probability\;percentage\;of\;class\;j\;in\;a\;node.$$
We split the dataset by selecting the class having the highest *Information Gain* or the lowest *Gini Index*.

***Advantages of the Decision Tree:***
- It can be very useful for solving decision-related problems.
- It helps to think about all the possible outcomes for a problem.
- There is less requirement of data cleaning compared to other algorithms.

***Disadvantages of the Decision Tree:***
- The decision tree contains lots of layers, which makes it complex.
- It may have an overfitting issue, which can be resolved using the *Random Forest algorithm*.
- For more class labels, the computational complexity of the decision tree may increase.

***Applications:***
1. *Marketing:* Businesses can use decision trees to enhance the accuracy of their promotional campaigns by observing the performance of their competitors’ products and services.
2. *Retention of Customers:* Companies use decision trees for customer retention through analyzing their behaviors and releasing new offers or products to suit those behaviors. By using decision tree models, companies can figure out the satisfaction levels of their customers as well.
3. *Random Forest Emsemble:* Decision trees can be combined and can be used with bagging technique for more accurate results with less errors. Example: Random Forest.
***
#### 9. Explain the concept of feature selection in machine learning.
***
**Ans :**
Feature Selection is a process of automatically or manually selecting the subset of most appropriate and useful features to be used in model building. The subset of most important features are selected from the original dataset by removing the redundant, irrelevant, and noisy features.

Machine learning models follow a simple rule: whatever goes in, comes out. If we put garbage (noise) into our model, we can expect the output to be garbage too. Further, having a lot of data can slow down the training process and cause the model to be slower. This is where feature selection helps us remove noise from our data and also reduce the training time.

Below are some benefits of using feature selection in machine learning:
- It helps in avoiding the curse of dimensionality.
- It helps in the simplification of the model so that it can be easily interpreted by other people.
- It reduces the training time which results in reduced computation usage.
- It reduces overfitting hence enhance the generalization.

***Feature Selection Techniques:***
Feature selection methods can be divided mainly into two categories:

1. ***Supervised Techniques:***
- **Wrapper Methods:** In wrapper methodology, selection of features is done by considering it as a search problem, in which different combinations are made, evaluated, and compared with other combinations. It trains the algorithm by using the subset of features iteratively. Examples include *Forward Selection*, *Backward Elimination*, *Recursive Feature Elimination*.
- **Filter Methods:** In Filter Method, features are selected on the basis of statistics measures. This method does not depend on the learning algorithm and chooses the features as a pre-processing step. The advantage of using filter methods is that it needs low computational time and does not overfit the data. Common techniques are *Information Gain*, *Chi-square Test*, *Fisher's Score*, *Missing Value Ratio*, etc.
- **Embedded Methods:** Embedded methods combined the advantages of both filter and wrapper methods by considering the interaction of features along with low computational cost. These are fast processing methods similar to the filter method but more accurate than the filter method. Examples are *Regularization*, *Random Forest Difference*.

2. ***Unsupervised Techniques:***
- These techniques can be used for unlabeled data. For Example- *K-Means Clustering*, *Principal Component Analysis*, *Hierarchical Clustering*, etc.
***
#### 10. Describe the logistic regression algorithm and its applications.
***
**Ans :**
Logistic regression is one of the most popular machine learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.
Logistic regression is a process of modeling the probability of a discrete outcome given input variables. 

The most common logistic regression models have a binary outcome; something that can take two values such as true/false, yes/no, and so on. For example, detecting whether an email is spam or not, whether a person can get a loan. Multinomial logistic regression can model scenarios where there are more than two possible discrete outcomes. 

***Advantages:***
- Easy to implement and interpret and efficient to train.
- Very fast at classifying unknown records.
- Performs well on linearly separable data and can be easily extended to multi-class classification problems.
- Robust to outliers.

In Logistic regression, we use a sigmoid function to map the predicted values between the range of 0 and 1. We use the concept of the threshold value, which defines the probability of either 0 or 1. Such as values above the threshold value tends to 1, and a value below the threshold values tends to 0. Using this method, we can categorize the target variable.

Let's consider the equation of a straight line:
$$\begin{aligned}y = mx+c \\\end{aligned}$$
We need to predict the Probability that a certain variable belongs to the class or not. Lets take the Logit(Logs-odd) function. When we fit the function to the line, we get:
$$log\left( \frac{P}{1-P} \right) = mx\;+\;c $$
$$\left( \frac{P}{1-P} \right) = e^{mx+c} $$
This is the equation for the logistic regression. If we solve for *P*, we get our *Logistic function*, also called as *Sigmoid function*.
$$ P=\frac{1}{1+e^{-(mx+c)}}$$
![[logistic_regression | 400]]

We use *Maximum Likelihood Estimator(MLE)* to estimate the model parameters. This estimation method is one of the most widely used. This selects the set of values of the model parameters that maximize the likelihood function.

***Type of Logistic Regression:***
Logistic Regression can be classified into three types:
- **Binomial:** In binomial Logistic regression, there can be only two possible class of the dependent variables, such as 0 or 1, Pass or Fail, etc.
- **Multinomial:** In multinomial Logistic regression, there can be 3 or more possible unordered classes of the dependent variable, such as "cat", "dogs", or "sheep"
- **Ordinal:** In ordinal Logistic regression, there can be 3 or more possible ordered classes of dependent variables, such as "Low", "Medium", or "High".

***Applications of Logistic Regression:***
- **Optical Character Recognition:** An optical character recognition (OCR) method, often called text recognition, may turn handwritten or printed characters into text that computers can understand. The output is categorical, which can be implemented with the help of logistic regression.
- **Fraud Detection:** Fraud detection is about detecting scams and preventing fraudsters from acquiring money or property by deception. We can use logistic regression to differentiate between fraudulent and legitimate transactions.
- **Email Spam Detection** : We can detect spam emails using Logistic Regression based on email attributes like word count, country of origin, suspicious words present, etc. and prevent phishing and frauds.
- **Disease prediction:** In medicine, this analytics approach can be used to predict the likelihood of disease or illness for a given population.
- **Churn prediction**: Specific behaviors may be indicative of churn in different functions of an organization. We can predict when customers will stop using a certain product or the employees at risk of leaving an organization.
***
#### 11. Explain the concept of support vector machines (SVM) in machine learning.
***
**Ans :**  
Support Vector Machines (SVMs) are a popular supervised machine learning algorithm used for classification and regression tasks. SVMs are particularly effective in solving complex problems with a clear margin of separation between classes.

The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

![[support-vector-machines|400]]

**Support Vectors:** The data points or vectors that are the closest to the hyperplane and which affect the position of the hyperplane are termed as Support Vectors.

**Margin:** It is the distance between the hyperplane and the observations closest to the hyperplane (support vectors). In SVM large margin is considered a good margin.

Let’s understand the working of SVM using an example. Suppose we have a dataset that has two classes (Green and Blue). We want to classify the new data point as either blue or green. Since we are plotting the data points in a 2-dimensional graph we call this decision boundary a **straight line** but if we have more dimensions, we call this decision boundary a **hyperplane**.

![[svm-working|400]]

The best hyperplane is that plane which has the maximum distance from both the classes, and this is the main aim of SVM. This is done by selecting the hyperplane having the maximum margin. In this example, we can see that *HP2* is a good decision boundary.

To classify a point as negative or positive we need to define a decision rule. We can define decision rule as:
$$\ y=\begin{cases}
+1  & \mathrm{if}\; \vec X.\vec w + b\geq0 \\
-1  & \mathrm{if}\; \vec X.\vec w + b < 0
\end{cases}$$
where vector *X* is our data point, vector *w* is the weight(normalized vector perpendicular to our hyperplane) and *b* is the offset value. Then we try to get  values of *(w,b)* such that we have the maximum possible margin denoted by *d*.

Now, the above formula works well if we have linearly separable data, but doesn't work well over non-linear data due to different constraints. To overcome this, we have various parameters to control the focus on Margin Maximization. Such models are known as Soft Margin SVM. Soft Margin has an extra function called *Zeta* which is 0 for correctly classified values and is non-zero for incorrectly classified values.

SVM also has different kernels which allow it to work extremely well with non-linearly separable datasets. Some of the kernel functions in SVM are as follows:
1. **Polynomial Kernel:** In the case of polynomial kernel, you also have to pass a value for the degree parameter *d* of the SVC class. This basically is the degree of the polynomial. *X1* and *X2* are two features.
$$ f(X1, X2) = (X1^T.X2+1)^d $$
2. **Gaussian Kernel:** The Gaussian kernel is an example of a radial basis function kernel. It can be represented with this equation:
$$ f(X1,X2) = e^{(-\gamma||X1 - X2||^2)}$$
3. **Sigmoid Kernel:** The equation for sigmoid kernel is
$$f(X1,X2)=tanh(\gamma.X1^TX2+r)$$

**Advantages of SVM:**
1. SVM works better when the data is Linear.
2. It is more effective in high dimensions.
3. With the help of the kernel trick, we can solve any complex problem.
4. SVM is not sensitive to outliers.
5. Can help us with better classification.

**Applications of Support Vector Machine:**
1. SVMs are helpful in text and hypertext categorization.
2. Classifications of images can also be performed using SVM. Experimental results show that SVMs achieve significantly higher search accuracy than other models.
3. SVM can also be used to implement Optical Character Recognition as it can help recognize text more precisely using optimized support vectors.


***
#### 12. Describe the k-means clustering algorithm and its applications.
***
**Ans :**
K-Means Clustering is an unsupervised learning algorithm that is used to solve the clustering problems in machine learning and data science. It groups the data into precisely *K* clusters, hence the name K-means clustering. It allows us to cluster the data into different groups and provides a convenient way to discover the categories in the unlabeled dataset on its own without the need for any training.

![[k-means-cluster|400]]

The K-means algorithm starts by randomly choosing a centroid value for each cluster. After that the algorithm iteratively performs the following three steps:
1. Find the Euclidean distance between each data point and centroids of all the clusters.
2. Assign the data points to the cluster whose centroid is the nearest to the point.
3. Calculate new centroid values by taking the mean of the coordinates of all the data points in each cluster.

To know the best value for *K*, we use an elbow curve method. For each *K*, calculate the total within-cluster sum of squares (WSS). This can be used to determine K.
- Perform K-means clustering with all different values of K. For each of the K values, we calculate average distances to the centroid across all data points.
- Plot these points and find the point where the average distance from the centroid falls suddenly (*Elbow*). The corresponding K value will be the best value for K.

***Advantages of k-means clustering:***
1. Relatively simple to implement.
2. Scales to large data sets.
3. Guarantees convergence.
4. Easily adapts to new examples.
5. Generalizes to clusters of different shapes and sizes.

***Applications:***
1. **Customer Insight:** Cluster analysis can help the retail chain get desired insights on customer demographics, purchase behavior, and demand patterns across locations.
2. **Marketing:** Cluster Analysis can be helpful in the field of marketing. It can help in market segmentation and positioning and identify test markets for new product development.
3. **Social Media:** In the areas of social networking and social media, Cluster Analysis is used to identify similar communities within larger groups.
4. **Medicine:** Cluster Analysis has also been widely used in the field of biology and medical science, like sequencing into gene families, human genetic clustering, building groups of genes, clustering of organisms at species, and so on.
***
#### 13. What is the difference between clustering and classification in machine learning?
**Ans :**

| ***Clustering***                                                                                                                                | ***Classification***                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Clustering is the process of grouping data into clusters such that data points in the same cluster are similar to each other.                   | Classification is a process of finding a function which helps in dividing the dataset into classes based on different parameters. |
| The task of the clustering algorithm is to group the provided data into a specified number of clusters.                                         | The task of the classification algorithm is to map the input variables $(x_i)$ with the discrete output variable (y).              |
| In clustering, there are no labels for training data.                                                                                           | Training data is labeled in classification.                                                                                       |
| Clustering is an unsupervised machine learning algorithm.                                                                                       | Classification is a supervised machine learning algorithm.                                                                        |
| Its objective is to group a set of objects to find whether there is any relationship between them.                                              | Its objective is to find which class a new object belongs to from the set of predefined classes.                                  |
| Clustering is used to make sense of existing data and find similarities between variables.                                                      | Classification is used to classify future data into already existing classes.                                                     |
| Clustering is an easy to implement and interpret algorithm.                                                                                     | Classification has a complex implementation and there are lots of parameters involved.                                            |
| Popular algorithms used for clustering include K-Means, Mean-Shift Clustering, and Density-Based Spatial Clustering. | Popular algorithms for classification include Naive Bayes Classifier, Decision Trees, and Random Forests.                         |
***
#### 14. Explain the concept of gradient descent in machine learning.
***
**Ans :**
Gradient Descent is used to train machine learning models by means of minimizing errors between actual and expected results. It is one of the most commonly used iterative optimization algorithms to train machine learning and deep learning models. It helps in finding the local minimum of a function, more specifically a loss function.

The main objective of using a gradient descent algorithm is to minimize the loss function. To achieve this goal, it performs two steps iteratively:
- Calculates the first-order derivative of the loss function to compute the gradient or slope of that function.
- Move away from the direction of the gradient, which means slope increased from the current point by alpha times, where alpha ($\alpha$) is defined as *Learning Rate*. It is a tuning parameter in the optimization process which helps to decide the length of the steps.

Let's take an example of gradient descent in Linear Regression with loss function as *Residual Sum of Squares*.

**Equation of a straight line:**
$$ y=mx+c$$
*x* : independent variable
*y* : dependent/target variable
*m* : slope of the line
*c* : y-intercept

**Residual Sum of Squares (Loss Function):**
$$R.S.S.=\sum_{i=1}^n(y_i - y_{i_{pred}})^2$$
Substituting the values of $y_{i_{pred}}=mx_i+c$ in the above equation, we get
$$ R.S.S.=\sum_{i=1}^n(y_i - (mx_i+c))^2$$
1. Initially let m = 0 and c = 0. Let *L* be our learning rate. This controls how much the value of *m* changes with each step. *L* could be a small value like 0.0001 for good accuracy.

2. Let's calculate the partial derivative of **R.S.S.** with respect to slope *m* and intercept *c* to get the derivative value $D_m$ and $D_c$. 
$$D_m= \sum_{i=1}^n2(y_i - (mx_i+c))(-x_i)$$
$$D_c= \sum_{i=1}^n2(y_i - (mx_i+c))$$
3. Now we update the current value of *m* and *c* using the following equation:
$$m = m- LearningRate*D_m$$
$$c=c-LearningRate*D_c$$
4. We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of *m* and *c* that we are left with now will be the optimum values.

Gradient Descent can be implemented for any type of model provided we have a proper loss function. It is extremely helpful in models where the loss function does not have a closed form solution. 

***Types of Gradient Descent:***
1. **Batch Gradient Descent:** Batch gradient descent (BGD) finds the error for each point in the training set and update the model after evaluating all training examples. In simple words, it is a greedy approach where we have to sum over all examples for each update.
2. **Stochastic Gradient Descent:** Stochastic gradient descent (SGD) runs one training example per iteration. It processes a training epoch for each example within a dataset and updates each training example's parameters one at a time. It is extremely helpful when dealing with large datasets.
3. **Minibatch Gradient Descent:** Mini Batch gradient descent is the combination of both batch gradient descent and stochastic gradient descent. It divides the training datasets into small batch sizes then performs the updates on those batches separately. This improves efficiency and provides good results in short time.
***
#### 15. Describe the Naive Bayes algorithm and its applications.
***
**Ans :**
A Naive Bayes classifier is a probabilistic machine learning model that is used for classification tasks. The crux of the classifier is based on the Bayes theorem. In simple terms, a Naive Bayes (NB) classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

- NB gives better result in multi class problems.
- NB gives better results when there is least collinearity among independent features.

For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple. Similarly, this classifier considers each feature as independent and ignores all relationships between variables and that is why it is known as *Naive*. 

Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, it is known to outperform even highly sophisticated classification methods. Bayes theorem provides a way of calculating posterior or conditional probability $P(c|x)$ from $P(c)$, $P(x)$ and $P(x|c)$. The following is the equation representing Bayes Theorem:

![[naive-bayes|400]]

Let's take an example of weather conditions. Let us apply Naive Bayes algorithm to find out that if the weather is sunny, then should the player play ?
**Weather condition table:**

| ***Weather*** | ***Yes*** | ***No*** |
| ----------- | ------- | ------ |
| Overcast    | 5       | 0      |
| Rainy       | 2       | 2      |
| Sunny       | 3       | 2      |
| Total       | 10      | 5      | 

**Likelihood table for weather condition:**

| ***Weather*** | ***Yes***     | ***No*** |             |
| ----------- | ----------- | ------ | ----------- |
| Overcast    | 5           | 0      | 5/14 = 0.35  |
| Rainy       | 2           | 2      | 4/14 = 0.29 |
| Sunny       | 3           | 2      | 5/14 = 0.35 |
| All         | 10/14 = 0.71 | 4/14 = 0.29       |             |

Let's calculate if the weather is sunny, then the player should play or not ?
Applying *Bayes Theorem*,
$$P (Yes \mid Sunny) = \frac{P(Sunny \mid Yes)*P(Yes)}{P(Sunny)}$$
$$P(Yes\mid Sunny) = (0.3*0.71)/0.35=0.60$$
Similarly,
$$P (No \mid Sunny) = \frac{P(Sunny \mid No)*P(No)}{P(Sunny)}$$
$$P(No\mid Sunny) = (0.5*0.29)/0.35=0.41$$

As $P(Yes \mid Sunny) > P(No \mid Sunny)$, hence on a sunny day, a player can play the game. Naive Bayes uses a similar method to predict the probability of different class based on various attributes.

***Advantages of Naive Bayes Classifier:***
- Naive Bayes is one of the fast and easy ML algorithms to predict a class of datasets.
- It can be used for Binary as well as Multi-class Classifications.
- It performs well in Multi-class predictions as compared to the other Algorithms.
- It is the most popular choice for *text classification problems*.

***Disadvantages of Naive Bayes Classifier:***
- Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.

***Applications:***
- It is used for Credit Scoring.
- It is used in medical data classification.
- It can be used in real-time predictions because Naïve Bayes Classifier is an eager learner.
- It is used in Text classification such as Spam filtering and Sentiment analysis.
***
#### 16. What is ensemble learning, and how is it used in machine learning?
***
**Ans :**
Ensemble learning is a machine learning approach in which the predictions from multiple models are combined to enhance the accuracy of the ultimate prediction. This method combines multiple machine learning models into one predictive model. This technique permits higher predictive performance with low bias and low variance.

Ensemble Learning tries to capture complementary information from its different contributing models. An ensemble framework is successful when the contributing models are statistically diverse. Each model in the ensemble method is known as a weak learner. The intuition is that when you combine several weak learners, they can become strong learners. Each weak learner is fitted on the training set and provides predictions. The final prediction result is computed by combining the results from all the weak learners.

***Ensemble Learning Techniques:*** There are multiple ensemble techniques but the following methods are often used.

1. **Stacking:** Stacking is the process of combining various models in order to reduce their biases. Predictions from each model is stacked together and used as input to a final model (usually called a _meta-model_) that computes the final prediction. Diversity comes from the different machine learning models used as ensemble members.

![[stacking-ensemble|400]]

Popular algorithms based on this approach include *Stacked Models (canonical stacking)*, *Blending*, *Super Ensemble*.

2. **Bagging:** The Bagging ensemble technique is the acronym for *Bootstrap Aggregating*. Random subsets of a dataset are created using replacement. These subsets are now treated as independent datasets, on which the same machine learning model will be fit. During test time, the predictions from all models trained are accounted for. 

![[bagging-ensemble|400]]

There is an aggregation mechanism used to compute the final prediction, like averaging, weighted averaging, etc. Examples of Bagging algorithms are *Random Forest Classifiers*, *Canonical Bagging*, *Extra Trees*, etc.

3. **Boosting:** Boosting ensemble works in a different way as compared to other methods. Here, instead of parallel processing of data, sequential processing of the dataset occurs. The first model is fed with the entire dataset, and the predictions are analyzed. The instances where Model-1 fails to produce correct predictions are fed to the second model. This helps the second model focus more on the data points where the first model couldn't predict well and improve upon the overall score.

![[boosting-ensemble|500]]

Boosting is a very popular ensemble method. Examples of boosting algorithms are *Adaboost (Adaptive Boosting)*, *XGBoost*, *Gradient Tree Boosting*, etc.

***Advantages of Ensemble Learning:*** 
1. **Performance:** An ensemble can make better predictions and achieve better performance than any single contributing model. They achieve better predictive results than a single predictive model.
2. **Robustness**: An ensemble reduces the spread or dispersion of the predictions and model performance.

***Disadvantages of Ensemble Learning:***
- The only disadvantage of using ensemble methods is that the computation is very intensive on large datasets and increases exponentially when more models are added. Although the advantages far outweigh the disadvantages and ensemble learning is now increasingly popular and useful.
***
#### 17. What is the difference between batch and online learning?
***
**Ans :**

| ***Batch Machine Learning***                                                                                                   | ***Online Machine Learning***                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batch learning, also known as offline learning, involves training a model on a fixed dataset, or a batch of data, all at once. | Online learning, also known as incremental learning or streaming learning, involves training a model on new data as it arrives, one observation at a time. |
| The model is trained on the entire dataset, and then used to make predictions on new data.                                     | The model is updated each time a new observation is received, allowing it to adapt to changes in the data over time.                                       |
| Batch learning requires a complete dataset before training of the model can begin.                                             | Online learning can be performed on changing data and the model can be updated once new data arrives.                                                    |
| Batch learning is commonly used in situations where the dataset is relatively small and can be processed quickly.              | Online learning is commonly used in situations where the data is too large to be processed all at once, or where the data is constantly changing.          |
| Batch learning is faster and requires less computational resources.                                                            | Online learning is typically a slow process and uses more computational resources due to model being trained constantly.                                   |
| Batch learning is not good at handling changing or large datasets.                                                             | Online learning is flexible and can quickly adapt to new data.                                                                                             |
| Model cannot be updated to new data without retraining the entire model.                                                       | Model can be updated by training on new data once it arrives.                                                                                              |
| Batch learning is less complex as there is only one system of data to be handled.                                              | Online learning is complex due to data being real-time and multiple systems needs to be in place.                                                          |
| Used in applications where data patterns remain constant and don’t have sudden concept drifts.                                 | Used in applications where new data patterns are constantly required.                                                                                      |
| Examples are Image Classification and anywhere where data remains constant for some time.                                      | Used in Finance, Economics, E-Commerce, Weather Prediction where analysis is need on current data.                                                         |
***
#### 18. What is the difference between supervised and unsupervised reinforcement learning?
***
**Ans :** 

| **No.** |                                                                    **Reinforcement Learning**                                                                    |
| ------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 1       |                                Reinforcement learning algorithms learn through trial-and-error interactions with an environment.                                 |
| 2       |                            Reinforcement learning agents receive feedback in the form of rewards or penalties based on their actions.                            |
| 3       |                         The goal of reinforcement learning is to learn an optimal policy that maximizes the cumulative reward over time.                         |
| 4       |                         Reinforcement learning involves an agent interacting with an environment and taking actions based on its policy.                         |
| 5       |                  The agent receives feedback signals in the form of rewards or punishments from the environment to guide its learning process.                   |
| 6       |                 Reinforcement learning involves sequential decision-making, where the agent learns by observing the consequences of its actions.                 |
| 7       | Reinforcement learning algorithms can be used in scenarios such as game playing, robotics, autonomous vehicles, recommendation systems, and resource management. |
| 8       |     Reinforcement learning algorithms include Q-Learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Actor-Critic methods, among others.     |

***(This answer below explains about supervised reinforcement learning and related process and concepts, not sure about the information being accurate.)***

**Supervised Reinforcement Learning:**
1. Supervised reinforcement learning, also known as apprenticeship learning or imitation learning, is a combination of supervised learning and reinforcement learning techniques.
2. It aims to leverage the benefits of both approaches to solve complex tasks.
3. In supervised reinforcement learning, an agent learns from both expert demonstrations and reinforcement signals. Here's how the process typically works:
	- *Supervised Learning Phase*: Initially, the agent is provided with a set of expert demonstrations where an expert or a human demonstrator performs the task. These demonstrations consist of input observations and corresponding optimal actions. The agent uses supervised learning techniques, such as classification or regression, to learn a mapping between the observations and the expert's actions.
	- *Reinforcement Learning Phase*: After the supervised learning phase, the agent transitions to the reinforcement learning phase. The agent interacts with the environment and receives reinforcement signals, typically in the form of rewards or penalties, based on its actions. The agent uses these reinforcement signals to further refine its learned policy.
	- *Policy Improvement*: The agent combines the learned policy from the supervised learning phase with the reinforcement signals obtained during the reinforcement learning phase. It iteratively improves the policy through trial-and-error exploration and exploitation, optimizing for higher cumulative rewards.

Supervised reinforcement learning can be beneficial in scenarios where expert demonstrations are available or when learning from scratch using only reinforcement signals is challenging. It allows the agent to leverage existing expertise to accelerate learning and potentially achieve better performance compared to starting from random exploration.
***
#### 19. Explain the bagging technique and its applications.
***
**Ans :**
**Bagging:** The Bagging ensemble technique is the acronym for *Bootstrap Aggregating*. For this method, subsamples from a dataset are created and they are called *bootstrap sampling*. Random subsets of a dataset are created using replacement. These subsets are now treated as independent datasets, on which the same machine learning model will be fit. During test time, the predictions from all models trained are accounted for. 

![[bagging-ensemble|400]]

***Implementation Steps of Bagging:***
- *Step 1:* Multiple subsets are created from the original data set with equal tuples, selecting observations with replacement.
- *Step 2:* A base model is created on each of these subsets.
- *Step 3:* Each model is learned in parallel with each training set and independent of each other.
- *Step 4:* The final predictions are determined by combining the predictions from all the models.

There is an aggregation mechanism used to compute the final prediction, like averaging, weighted averaging, etc. Examples of Bagging algorithms are *Random Forest Classifiers*, *Canonical Bagging*, *Extra Trees*, etc.

***Advantages:***
1. The bagging technique reduces model over-fitting. 
2. It also performs well on high-dimensional data. Moreover, the missing values in the dataset do not affect the performance of the algorithm.
3. Bagging can reduce the variance within a learning algorithm. This is particularly helpful with high-dimensional data, where missing values can lead to higher variance.

***Applications:***
Bagging technique is used in a variety of applications. One main advantage is that it reduces the variance in prediction by generating additional data whilst applying different combinations and repetitions. Below are some applications where the bagging algorithm is widely used:
- *Banking Sector*
- *Medical Data Predictions:* By training multiple models on different subsets of medical data, bagging can improve the overall accuracy and reliability of the predictions.
- *High-dimensional data:* It helps to mitigate the curse of dimensionality and improve generalization by combining multiple models trained on different subsets of features.
- *Land cover mapping:* Bagging can enhance the accuracy of land cover classification by reducing variance and bias using multiple learning models.
- *Fraud detection:* Bagging can be utilized in fraud detection systems to identify and classify suspicious activities in financial transactions.
- *Network Intrusion Detection Systems*
***
#### 20. Describe the principal component analysis (PCA) algorithm and its applications.
***
**Ans :**
*Principal Component Analysis (PCA)* is an unsupervised learning algorithm that is used for dimensionality reduction in machine learning or for visualization of high-dimensional datasets. It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation. It is one of the popular tools that is used for exploratory data analysis and predictive modeling. 

**Principal Components (PC):** The transformed new features or the output of PCA are the Principal Components. The number of these PCs are either equal to or less than the original features present in the dataset. They are a linear combination of the original features. The importance of each component decreases when going to 1 to n, it means that *PC-1* has the most importance, and *nth PC* will have the least importance.

***Steps for PCA Algorithm:***
1. **Standardizing the data:** If the features in the dataset have different scales, it is important to standardize them to have zero mean and unit variance. This ensures that each feature contributes equally to the PCA.
2. **Calculating the Covariance Matrix:** The next step is to calculate the covariance matrix for the data. This matrix represents the relationships and variances between each features in the dataset.
3. **Calculating the Eigenvalues and Eigenvectors:** Calculate the eigenvalues and eigenvectors for the covariance matrix. Eigenvectors of the covariance matrix are the directions of the axes with high information. And the coefficients of these eigenvectors are the eigenvalues.
4. **Calculating the Principal Components:** Sort the eigenvalues in decreasing order and select the corresponding eigenvectors. These eigenvectors become the principal components. Each principal component will be a linear combination of the original features and all the components will be independent of each other. 
5. **Transform the data:** We can project the original data onto the selected principal components to obtain the lower-dimensional representation of the data.

***Applications of PCA:**
1. **Dimensionality Reduction**: PCA is a popular technique used for dimensionality reduction, which is the process of reducing the number of variables in a dataset. This helps avoid the curse of dimensionality.
2. **Feature Selection**: PCA can be used for feature selection, which is the process of selecting the most important variables in a dataset. This is useful in machine learning, where the number of variables can be very large, and it is difficult to identify the most important variables.
3. **Data Visualization**: PCA can be used for data visualization. By reducing the number of variables, PCA can plot high-dimensional data in two or three dimensions, making it easier to interpret.
4. **Noise Reduction**: PCA can be used to reduce the noise in data. By removing the principal components with low variance, which are assumed to represent noise, the data quality can be improved.

***Disadvantages of Principal Component Analysis:***
1. **Interpretation of Principal Components**: The principal components created by PCA are linear combinations of the original variables, and it is often difficult to interpret them in terms of the original variables. This can make it difficult to explain the results of PCA to others.
2. **Data Scaling**: PCA is sensitive to the scale of the data. If the data is not properly scaled, then PCA may not work well. Therefore, it is important to scale the data before applying PCA.
3. **Information Loss**: PCA can result in information loss. While PCA reduces the number of variables, it can also lead to loss of information. The degree of information loss depends on the number of principal components selected.
4. **Non-linear Relationships**: PCA assumes that the relationships between variables are linear. However, if there are non-linear relationships between variables, PCA may not work well.
5. **Computational Complexity**: Computing PCA can be computationally expensive for large datasets. This is especially true if the number of variables in the dataset is large.
***
#### 21. Explain the difference between a decision tree and a random forest algorithm.
***
**Ans :**

| ***Decision Tree***                                                                                                                                                                    | ***Random Forest***                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A decision tree is a supervised learning algorithm that uses a hierarchical structure to make predictions by recursively splitting the data based on feature values.                   | Random Forest is a tree-based machine learning algorithm that leverages the power of multiple decision trees for making decisions.                                                  |
| It creates a tree-like model where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome or prediction.            | It builds a collection of decision trees, where each tree is trained on a random subset of the training data (bootstrap sampling) and a random subset of features.                  |
| In decision trees, there is a high possibility of overfitting, especially with complex or noisy data                                                                                   | Random forests have a reduced risk of overfitting the data by aggregating the output from multiple decision trees to predict a result.                                              |
| Decision trees can be sensitive to training data and may not perform well on unseen data if the training data is not proper.                                                           | Random Forest can handle missing data and maintain good performance even if some features have missing values.                                                                      |
| Less computational resources are needed to model a decision tree.                                                                                                                      | Random Forest requires more computational resources due to presence of multiple decision trees, which needs more training.                                                          |
| Decision trees are easy to interpret and visualize, making them useful for understanding the decision-making process.                                                                  | The major drawback of the random forest model is its complicated structure due to the grouping of multiple decision trees.                                                          |
| Decision trees can be implemented quickly and they can operate fast on large datasets.                                                                                                 | The random forest model needs rigorous training and consumes more time with increase in size of the dataset.                                                                        |
| Decision Trees can suffer from high variance, meaning that small changes in the training data can lead to different decision tree structures and, consequently, different predictions. | Random Forest mitigates the problem of high variance by training each tree on a random subset of the training data.                                                                 |
| Decision Trees are often used when interpretability and simplicity are important, as they provide a clear decision-making structure.                                                   | Random Forest is preferred when higher prediction accuracy, reduced variance, and handling complex datasets are desired. Examples include Finance, Healthcare, Remote Sensing, etc. |
***
#### 22. What is Recursive Feature Elimination (RFE)? How can it be used in Machine Learning?
***
**Ans :**
*Recursive Feature Elimination (RFE)*, as the name suggests, is a feature elimination technique that ranks the features based on their importance and returns top _n_ features after eliminating the least important features, where _n_ is given by the user.

***How RFE Algorithm works:***
1. RFE takes a supervised learning estimator that was already fit using data with all features.
2. It then considers the coefficient associated with each feature. The value of these coefficients represents their importance with the target variable. Feature with the least coefficient value is considered as the least important and so on.
3. It then eliminates the least important feature and rebuilds the model with the remaining set of features. The number of features to be dropped at each iteration can also be specified by the user. However, it is preferred to eliminate 1 feature at a time since we build a new model after the elimination.
4. At each iteration, it rebuilds the model and eliminates the least important feature(s) and repeats the process until it is left with *n* features. After that, it ranks features based on their time of elimination.

***RFE in Machine Learning:***
Recursive Feature Elimination helps in feature selection and can offer several benefits:
- **Improved Model Performance**: By eliminating irrelevant or redundant features, RFE can improve the model's performance. Removing irrelevant features reduces noise and focuses the model on the most informative features, leading to better generalization.  
- **Reduced Overfitting**: RFE can reduce overfitting by discarding features that may cause the model to learn noise or capture irrelevant patterns in the training data. 
- **Interpretability**: RFE can enhance the interpretability of the model by selecting a smaller set of features that are most relevant for prediction. It helps to identify the key variables that contribute to the model's decision-making process. 
- **Computational Efficiency**: By reducing the feature set, RFE can speed up the training process, especially when dealing with high-dimensional data where the number of features is large.

For more enhanced results, RFE can be combined with  other methods like _p-value_ for checking the significance level of the feature and *Variance Inflation Factor (VIF)* for multicollinearity check.
***
#### 23. What is Variance Inflation Factor (VIF)? How can it be used to address multicollinearity in Machine Learning?
***
**Ans :**
Variance Inflation Factor (VIF) is a statistical measure that quantifies the severity of multicollinearity in a dataset with multiple predictor variables. In simple words, Variance Inflation Factor measures how much the behavior (variance) of an independent variable is influenced, by its interaction/correlation with the other independent variables.

VIF helps detect multicollinearity in our training data. Multicollinearity occurs when there are two or more independent variables in a multiple regression model, which have a high correlation among themselves. When some features are highly correlated, we might have difficulty in distinguishing between their individual effects on the dependent variable.

***How VIF helps detecting multicollinearity:***
In VIF method, we pick each feature and regress it against all of the other features. For each regression, the factor is calculated as
$$VIF_{i} = \frac {1}{1-R_i^2}$$
Where, $R_i^2$ is the R-squared value obtained by regressing the $i^{th}$ predictor on the remaining predictors. Also called as the coefficient of determination in linear regression. Its value lies between 0 and 1.

As we see from the formula, greater the value of R-squared, greater is the VIF. Hence, greater VIF denotes greater correlation. This is in agreement with the fact that a higher R-squared value denotes a stronger collinearity. Generally, a VIF above 5 indicates a high multicollinearity, but the ultimate values depend on the context and the interpretation of the data.

***Implementing VIF:***
To implement VIF when working with data, we can either program the algorithm manually or use some packages. One such package is *statsmodels* which provides a function named *variance_inflation_factor()* for calculating VIF.

**Syntax** : `variance_inflation_factor(exog, exog_idx)`

**Parameters** : 
- *exog :*  an array containing features on which linear regression is performed.
- *exog_idx :*  index of the additional feature whose influence on the other features is to be measured.

There are multiple methods to check for multicollinearity like the correlation matrix, scatter plots, etc. but their findings only show the bivariate relationship between the independent variables. VIF is preferred as it can show the correlation of a variable with a group of other variables.

By examining VIF values for each independent variable in a regression model, researchers can identify and address potential multicollinearity, ensuring the reliability and interpretability of the regression analysis.
***
#### 24. What is the ROC curve? How is it used in evaluating classification models?
***
**Ans :***
The *Receiver Operator Characteristic (ROC)* curve is an evaluation metric for binary classification problems. It is a probability curve that plots the *True Positive Rate* against *False Positive Rate* at various threshold values and essentially separates the signal from the noise. In other words, it shows the performance of a classification model at all classification thresholds.

The True Positive Rate (TPR), also known as sensitivity or recall, represents the proportion of actual positive cases correctly classified as positive. The False Positive Rate (FPR), on the other hand, represents the proportion of actual negative cases incorrectly classified as positive. 

***Constructing an ROC curve:***
1. The classification model produces probability scores or confidence scores for each instance in the dataset.
2. By varying the classification threshold (decision threshold), the model's predictions are classified as either positive or negative. This threshold determines the balance between the TPR and FPR.  
3. Starting from a threshold that classifies all instances as positive, the threshold is gradually lowered, classifying more instances as negative. At each threshold, the TPR and FPR values are calculated.
4. The TPR is plotted on the y-axis, and the FPR is plotted on the x-axis, resulting in a curve that starts from the coordinate (0,0) and ends at (1,1).  
5. The curve from (0,0) to (1,1) represents the performance of our classifier. 

![[Pasted image 20230609194230.png|400]]

***Interpretation of ROC curve:***
A classification model with high classification power will have an ROC curve that is closer to the top-left corner, indicating a high TPR and a low FPR across different thresholds. An ideal classifier would have an ROC curve that coincides with the top-left corner, indicating a TPR of 1 and an FPR of 0.

ROC can be combined with *Area under the curve (AUC)* to measure the model's performance. For example, a model with AUC of approximately 0.5 would be no better than random predictions. An ideal classification model would have an AUC value of 1. AUC also helps comparing different model based on their respective AUC. For example, a model with 0.9 AUC would perform better than a model with 0.8 AUC value. Generally, an AUC value of 0.9 or higher is considered excellent.

The ROC curve provides a comprehensive view of the model's performance across different classification thresholds, allowing for a trade-off analysis between sensitivity and specificity. It makes evaluating classification models easier by enabling the comparison of different models and helps in selecting the optimal threshold based on the desired balance between true positives and false positives.
***
#### 25. What is precision and recall in Machine Learning? How are they used in evaluating classification models?
***
**Ans :**
Precision and recall are two commonly used metrics in evaluating the performance of a classification model, particularly in binary classification problems. They provide insights into the model's ability to make accurate positive predictions (precision) and capture all positive instances (recall).

**Precision:** It is the measure of how many of the positive predictions made by the model are actually correct. It focuses on the accuracy of the positive predictions. Mathematically, it is represented as:
$$Precision = \frac{True\;Positive\;(TP)}{True\;Positive\;(TP) + False\;Positive\;(FP)}$$
- True Positives (TP) represents the number of instances that are correctly predicted as positive.
- False Positives (FP) represents the number of instances that are incorrectly predicted as positive when they are actually negative.

Precision indicates the model's ability to avoid false positives. A higher precision value indicates that the model has a lower rate of falsely classifying negative instances as positive.

**Recall:** Recall, also known as sensitivity or true positive rate, is the measure of how well the model captures all the positive instances in the dataset. It focuses on the ability of the model to correctly identify positive instances. Mathematically, recall is calculated as:
$$Recall=\frac {True\;Positive\;(TP) }{True\;Positive\;(TP) + False\;Negative\;(FN)}$$
- False Negatives (FN) represents the number of instances that are incorrectly predicted as negative when they are actually positive.

Recall indicates the model's ability to avoid false negatives. A higher recall value indicates that the model has a lower rate of falsely classifying positive instances as negative.

**Precision-Recall Tradeoff**: Precision and recall are often inversely related. Increasing the classification threshold generally leads to higher precision but lower recall, and vice versa. The choice of threshold depends on the specific requirements of the problem at hand. 

For example, If our main focus is to detect a person having heart illness, our model should have high recall, meaning that it is better to classify a healthy person as positive rather than classifying a sick person as negative, which means we have to lower the False Negatives. 

Similarly, if we take the example of detecting a mail as spam, our model needs to have high precision which in turn means less recall score, as we need to lower False Positives. We would rather prefer a spam email (positive) being classified as not spam (negative), rather than a normal email (negative) being classified as spam (positive).
***
#### 26. Explain the difference between parametric and non-parametric Machine Learning algorithms.
***
**Ans :**

| ***Parametric Machine Learning***                                                                                                      | ***Non-Parametric Machine Learning***                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Algorithms that simplify the mapping function to a known form are called parametric machine learning algorithms.                               | Algorithms that do not make strong assumptions about the form of the mapping function are called nonparametric machine learning algorithms. |
| No matter how much data is given as input to a parametric model, it won’t change its mind about how many parameters it needs.          | By not making assumptions, non-parametric models are free to learn any functional form from the training data.                                               |
| Parametric models require the prior specification of a set of parameters that define the underlying distribution of the data.          | Non-parametric models do not rely on predefined parameter settings, and adapt to the data as they train.                                    |
| In parametric approach, if the predictor function is different than the true function, the resultant model will not fit the data well. | Non-parametric models have the potential to accurately fit a wider range of possible shapes for the actual or true function.                |
| Parametric models are much easier to fit because they only require the estimation of a set of parameters.                              | In the case of a non-parametric model, one needs to estimate some arbitrary function which is a much more difficult task.                   |
| Parametric models often have low accuracy but high interpretability.                                                                   | Non-parametric models have high accuracy but are not easily interpretable.                                                                  |
| Parametric models are efficient for large datasets with limited features.                                                              | These models perform well on high-dimensional data.                                                                                         |
| Examples of parametric models are Logistic Regression, Linear Discriminant Analysis (LDA), Perceptron, etc.                            | Examples of non-parametric models are Decision Tree, Support Vector Machines, etc.                                                          |
***
#### 27. What is the curse of dimensionality in Machine Learning? How can it be addressed?
***
**Ans :**
The *curse of dimensionality* refers to the challenges and issues that arise when working with high-dimensional data in machine learning. It refers to the fact that many traditional machine learning algorithms struggle or fail to perform well as the number of input features (dimensions) increases. Some of the difficulties that come with high dimensional data manifest during analyzing or visualizing the data, and some manifest while training the machine learning models.

***Why does it occur:***
1. **Sparsity of Data**: As the number of dimensions increases, the available data becomes sparse. In high-dimensional spaces, the data points are spread out, making it difficult to gather enough representative samples for accurate modeling and decision-making.
2. **Increased Computational Complexity**: High-dimensional data requires more computational resources and time to process, train models, and make predictions. The increased complexity can lead to slower training and inference times.
3. **Increased Model Complexity**: In high-dimensional spaces, the complexity of models tends to increase to capture the intricate relationships among features.

***Addressing Curse of Dimensionality:***
To address the curse of dimensionality, several techniques can be employed:
1. **Feature Selection**: Selecting relevant features can reduce the dimensionality of the data and focus on the most informative attributes. Feature selection methods help identify and retain the most important features for modeling.
2. **Dimensionality Reduction**: Techniques such as Principal Component Analysis (PCA) can reduce the dimensionality of the data by transforming it into a lower-dimensional space while preserving important patterns and structures. These methods help capture the most significant variations in the data.
3. **Regularization**: Regularization techniques, such as L1 and L2 regularization, can help control the complexity of models and prevent overfitting. By introducing penalty terms, these techniques encourage models to focus on the most important features and avoid relying heavily on less informative ones.
4. **Ensemble Methods**: Ensemble methods, like Random Forests, can remove the curse of dimensionality by combining multiple models. By leveraging the diversity of different models, ensemble methods can capture complex relationships even in high-dimensional spaces.

There are multiple methods to address problems with high dimensions, but selecting the best ones depend on the context of the problem and the structure and type of the data.
***
#### 28. What is an outlier in Machine Learning? How can it be detected and handled?
***
**Ans :**
An *Outlier* is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. Outliers are also called aberrations, abnormal points, anomalies, etc. Outliers can occur due to various reasons, such as measurement errors, data corruption, or genuinely rare events. Detecting and handling outliers is crucial because they can adversely impact the performance and accuracy of machine learning models.

***Methods to detect outliers:***
There are multiple methods to detect outliers. Some popular methods are as follows:
1. **Percentile Method:**
  - In this method, We first define the upper and lower bounds of a dataset using the desired percentiles. For example, we may use the 5th and 95th percentile for a dataset's lower and upper bounds respectively. Any observations or data points that reside beyond and outside of these bounds can be considered outliers.
2. **Inter-Quartile Range (IQR) method:**
  - The Interquartile range (IQR) is a measure of the dispersion of a dataset. The IQR is calculated by subtracting the first quartile from the third quartile of the dataset.
  - Using IQR, we can define a dataset's upper and lower bounds. The upper bound is defined as $Q3 + 1.5 * IQR$, and the lower bound is defined as $Q1 - 1.5*IQR$. Any observations or data points that reside beyond and outside these bounds can be considered outliers.
3. **Z-score method:**
  - For a given value, the respective z-score represents its distance in terms of the standard deviation. For example, a z-score of 2 represents that the data point is 2 standard deviations away from the mean. 
  - To detect the outliers using the z-score, we can define the lower and upper bounds of the dataset. Suppose upper and lower bound are defined as z = 3 and z = -3 respectively. This means any value more than 3 standard deviations away from the mean will be considered an outlier.
4. **Visualization Techniques**:
  - Visualizing the data can help identify outliers. Box plots, scatter plots, and histograms can reveal data points that are far from the majority of the distribution or exhibit unusual patterns. Combining multiple methods can help detect outliers with more precision.

***Handling Outliers:***
1. **Removal of outliers:** In some cases, it may be appropriate to simply remove the observations that contain outliers. This can be particularly useful if our data has a large number of observations and the outliers are not true representatives of the underlying population.
2. **Transform outliers:**  The impact of outliers can be reduced or eliminated by transforming the feature. For example, a log transformation of a feature can reduce the skewness in the data, reducing the impact of outliers.
3. **Impute outliers:** In this case, outliers are simply considered as missing values. Various imputation techniques for missing values, such as mean, median, mode, nearest neighbor, etc., can be used to replace the values for outliers.
4. **Using robust ML methods:** Some of the machine learning methods are less sensitive to outliers and can provide more reliable results when outliers are present in the data. Robust models such as robust regression or decision trees, can be used. These algorithms are designed to handle outliers more effectively by reducing their influence on the model's learning process.
***
#### 29. Explain the concept of feature scaling in Machine Learning.
***
**Ans :**
*Feature scaling* is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step. 

***Need for feature scaling:***
- Some machine algorithms are sensitive to the magnitude of the data, some are not. Hence, while working with models that are sensitive to magnitude, it is a good practice to scale the data before training the model to avoid poor performance.
- Without scaling, the machine learning algorithm may be biased towards features that have a higher magnitude in comparison with other features. After scaling, every feature is brought in the same range, and the model trains on every feature without bias.
- Scaling the features can improve the optimization process by making the flow of gradient descent smoother and helping to find the minimum of the cost function sooner, reducing time and computational usage.

***Types of feature scaling techniques:***
The following two methods are most commonly used for scaling the data.
1. **Normalization (Min-Max Scaling):**
  - Normalization is a data preprocessing technique used to adjust the values of features in a dataset to a common scale.
  - In this method, values are shifted and rescaled so that they end up ranging between 0 and 1.
  - The formula for Min-Max scaling is as follows: $$ X' = \frac {X-X_{min}}{X_{max}-X_{min}}$$
   Here, $X_{max}$ and $X_{min}$ are the maximum and the minimum values of the feature, respectively.
  - Normalization is sensitive to outliers but retains the original shape of the distribution.

2. **Standardization:**
  - Standardization is another scaling method where the values are centered around the mean with a unit standard deviation.
  - This means that after applying the scaling method, the mean of the attribute becomes zero, and the resultant distribution has a unit standard deviation.
  - This is the formula for standard scaling: $$X'=\frac {X-\mu} {\sigma}$$
   Here, $\mu$ and $\sigma$ are the mean and the standard deviation of the features respectively.
  - Standard scaling is less sensitive to outliers but changes the shape of the original distribution.
***
#### 30. What is the difference between a validation set and a test set in Machine Learning?
***
**Ans :**

| ***Validation Set***                                                                                                                                                                         | ***Test Set***                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| The validation set is generally a subset of the training data and is considered a part of the training of the model.                                                                         | Test set is independent of the training set but has a similar structure of features and is used as a benchmark to evaluate the model. |
| The model only sees this data for evaluation during training but does not learn from this data, providing an objective unbiased evaluation of the model.                                     | This data is used after the model has completed the training phase and is helpful in evaluating the model.                                     |
| This data is usually dynamic subset of the training data and accounts for 5-10% of the entire dataset.                                                                                       | This data is fixed before the model enters the training phase and accounts for 20-30% of the entire dataset.                                   |
| This set is used for hyper-parameter tuning and optimal model selection. It helps to find the best values for hyper-parameters.                                                              | Test set is used to evaluate the model to know how well it performs on new and unseen data.                                                    |
| Validation set is utilized in the training phase of machine learning.                                                                                                                        | Test set is utilized in the final phase of machine learning before deployment of model.                                                        |
| Validation set changes randomly and is combined with multiple models with different hyper-parameter values in order to find the optimal values that will give the best possible performance. | A test set is fixed and does not change during or after the training of the model.                                                             |
| Example: Validation set is used in combination with cross validation methods to find the best features and hyper-parameters.                                                                 | Test set is used to evaluate the training and test scores to find the accuracy of the model and to check for underfitting or overfitting.      |
***
#### 31. What are the steps involved in a typical Machine Learning project?
***
**Ans :**
A typical machine learning project involves seven major steps as follows:

1. **Gathering Data:** Data Gathering is the first step of the machine learning process. The goal of this step is to identify and obtain all data-related problems. In this step, we need to identify the different data sources, as data can be collected from various sources such as files, database, internet, or mobile devices. By performing the task of collecting, organizing and storing data, we get a coherent set of data, also called as a *dataset*. It will be used in further steps.
2. **Data Preparation:** Data preparation is a step where we put our data into a suitable place and prepare it to use in our machine learning training. In this step, first, we put all data together, and then randomize the ordering of data. This step can be further divided into data exploration, where we try to understand the nature, features and format of the data and data preprocessing.
3. **Data Wrangling:** Data wrangling is the process of cleaning and converting raw data into a useable format. It is the process of cleaning the data, selecting the variable to use, and transforming the data in a proper format to make it more suitable for analysis in the next step. It is one of the most important steps of the complete process. Cleaning of data is required to address the quality issues. We handle duplicate, missing values, etc. in this process.
4. **Data Analysis:** Now the cleaned and prepared data is passed on to the analysis step. This step involves Selection of analytical techniques, Building models, Review the result, etc. This step starts with the determination of the type of the problems, where we select the machine learning techniques such as *Classification*, *Regression*, *Cluster analysis*, *Association*, etc. then build the model using prepared data, and evaluate the model.
5. **Train Model:** Now the next step is to train the model, in this step we train our model to improve its performance for better outcome of the problem. We use datasets to train the model using various machine learning algorithms. Training a model is required so that it can understand the various patterns, rules, and features.
6. **Test Model:** Once our machine learning model has been trained on a given dataset, then we test the model. In this step, we check for the accuracy of our model by providing a test dataset to it. Testing the model determines the percentage accuracy of the model as per the requirement of project or problem.
7. **Deployment of the model:** The last step of machine learning life cycle is deployment, where we deploy the model in the real-world system. If the above-prepared model is producing an accurate result as per our requirement with acceptable speed, then we deploy the model in the real system. It is then accessed and utilized for the purpose for which it was created. Example: Classification of images, Price Prediction, Disease detection, etc.
***
#### 32. What is the difference between a linear and nonlinear regression in Machine Learning?
***
**Ans :**

| ***Linear Regression***                                                                                                           | ***Non-Linear Regression***                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Linear regression tries to map a linear relation between the independent and dependent variable, also called as line of best fit. | Non-linear Regression algorithms, model a non-linear relationship between the dependent and independent variables.           |
| The formula for linear regression is $y = \beta_0+\beta_1x_1+\beta_2x_2+...+\beta_nx_n$.                                          | One of the formula for non-linear regression is $y = f(X,\beta)+\epsilon$ , where $\epsilon$ is the error term.              |
| There is closed form solution for optimization of the loss function with a few exceptions.                                        | There is no closed form solution for optimizing the loss function and other iterative approaches are used.                   |
| Straight line of regression is fitted among the data to predict the new data points based on the existing data.                   | A polynomial curve of degrees *k* is fitted among the data points based on the existing data.                                |
| Linear regression is used for data that is linearly separable.                                                                    | Non-linear regression can help predict on non-linear data i.e. data that cannot be separated perfectly.                      |
| Linear model is restricted to fit a line of best fit on the dataset.                                                              | Non-linear regression is much more flexible in the shapes of the curves that it can fit.                                     |
| Linear regression is easier to use, simpler to interpret, and we obtain more statistics that help assess the model.               | Non-linear regression requires more effort both to find the best fit and to interpret the role of the independent variables. |
| Linear regression can be used to predict Height based on Weight, House Prices, etc.                                               | Non-linear regression can be used to predict exponential growth, inflation, currency depreciation, etc.                                                                                                                              |
***
#### 33. What is the purpose of dimensionality reduction in Machine Learning?
***
**Ans :**
*Dimensionality reduction* is the process of reducing the number of features (or dimensions) in a dataset while retaining as much information as possible. There are several techniques for dimensionality reduction, including principal component analysis (PCA), singular value decomposition (SVD), and linear discriminant analysis (LDA) and many more methods.

***Purpose of dimensionality reduction:***
There are several purposes of performing dimensionality reduction in machine learning:
1. **Curse of Dimensionality**: High-dimensional data can suffer from the curse of dimensionality, where the data becomes sparse and scattered in the feature space. Dimensionality reduction helps mitigate the curse of dimensionality by reducing the number of features and improving the efficiency and effectiveness of learning algorithms.
2. **Improved Model Performance**: Reducing the number of irrelevant or redundant features can improve the performance of machine learning models. By removing these factors, dimensionality reduction can enhance model accuracy, reduce overfitting, and improve generalization to unseen data.
3. **Computational Efficiency**: High-dimensional data requires more computational resources and time for training and inference. By reducing the dimensionality, the computational complexity of machine learning algorithms can be significantly reduced, allowing for faster model training, evaluation, and prediction. This is especially important for large-scale datasets or real-time applications where efficiency is crucial.
4. **Visualization and Interpretability**: Dimensionality reduction techniques can help visualize and interpret complex datasets. By mapping high-dimensional data to lower-dimensional spaces, it becomes easier to visualize and understand the patterns, relationships, and clusters within the data. We can use methods like PCA to map high dimensional data to 2D or 3D for better understanding of the data.
5. **Noise and Outlier Reduction**: Dimensionality reduction can help remove the effects of noisy or irrelevant features. This results in more robust and reliable models. It can also help identify and handle outliers, as they tend to become more apparent in lower-dimensional spaces.
6. **Reduce overfitting:** Dimensionality reduction finds a lower number of variables or removes the least important variables from the model. That will reduce the model’s complexity and also remove some noise in the data, which in turn reduces the probability of the model overfitting the data.
7. **Multicollinearity:** When we apply dimensionality reduction, it takes advantage of existing multicollinearity between the variables. For example, PCA combines highly correlated variables into a new set of uncorrelated variables. This can automatically reduce multicollinearity in your data.
***
#### 34. What is the difference between overfitting and underfitting in Machine Learning?
***
**Ans :**

| ***Overfitting***                                                                                                                               | ***Underfitting***                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Overfitting is when a model tries to fit the training data entirely and ends up memorizing the data patterns and the noise/random fluctuations. | Underfitting is when the model cannot create a mapping i.e. cannot create a relationship between the input and the target variable. |
| Overfitted models fail to generalize and do not perform well in the case of unseen data scenarios.                                              | Underfitted models perform poorly with high error on training as well as test data.                                                 |
| Overfitted models have high training accuracy but low validation/test accuracy.                                                                 | Underfitted models have both low training accuracy and low validation/test accuracy.                                                    |
| Overfitting is detected in the test phase where there is a high difference in training and testing accuracy scores, generally more than 5%.      | Underfitting is detected when the training error is very high and the model is unable to learn from the training data.              |
| Low bias and High variance are common indicators for overfitting.                                                                               | High bias and low variance are the most common indicators of underfitting.                                                          |
| Common reasons for overfitting are high model complexity, high noise and variance in data, insufficient training data, etc.                          | Common reasons for underfitting is simple model for complex data, high bias in data, incorrect hyper-parameter tuning, etc.         |
| Overfitting can be avoided by increasing the size of training data, feature selection, cross validation, L1 L2 regularization etc.              | Underfitting can be avoided by feature engineering, data augmentation, less regularization, more time for training etc.             |
***
#### 35. What is the purpose of hyperparameter tuning in Machine Learning?
***
**Ans :**
*Hyper-parameter tuning* is the process of selecting the best hyperparameters to use to build a machine learning model, and this tuning process is also known as hyperparameter optimization. Hyperparameters are parameters that are not learned from the data but are set prior to the model training process. They control the behavior and configuration of the learning algorithm, affecting the model's performance, complexity, and generalization ability.

Some common hyper-parameters are train-test-split ratio, batch size, number of epochs, numbers of clusters, learning rate, branches in decision trees, etc.

***Purpose of Hyper-Parameter tuning:***
1. **Model Performance Optimization**: Hyperparameters directly influence the model's performance metrics such as accuracy, precision, recall, F1 score, or mean squared error. By tuning the hyperparameters, you can search for the combination that maximizes the desired performance metric.
2. **Avoiding Overfitting or Underfitting**: Hyperparameters can control the complexity of the model, such as the depth of a decision tree or the number of hidden layers in a neural network. By tuning these hyperparameters, you can prevent overfitting (excessive model complexity) or underfitting (insufficient model complexity) and achieve a good balance.
3. **Generalization to Unseen Data**: Hyperparameters affect the model's ability to generalize well to new, unseen data. By tuning the hyperparameters, you can improve the model's generalization performance, allowing it to make accurate predictions on unseen data.
4. **Efficient Resource Utilization**: Hyperparameters can impact the computational resources required for training and inference. Tuning the hyperparameters can help find the right balance between model performance and computational efficiency, ensuring optimal resource utilization.
5. **Domain-Specific Considerations**: Different datasets and problem domains may require specific hyperparameter settings. By tuning the hyperparameters, you can adapt the model to the specific characteristics of the data and domain knowledge, improving its performance in the context of the problem.

The overall purpose of hyper-parameter tuning is to enhance the performance of the model and to save computational resources. Hyper-parameter tuning can be done in a number of ways. Some commonly used methods are GridSearchCV, RandomizedSearchCV, Bayesian Optimization, etc.
***
#### 36. Explain the difference between a simple and a multiple linear regression in Machine Learning.
***
**Ans :**

| ***Simple Linear Regression***                                                                                                                                       | ***Multiple Linear Regression***                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simple Linear Regression has one independent (predictor) and one dependent (target) variable.                                                                        | Multiple Linear Regression has more than one independent (predictors) and one dependent (target) variable.                                                                |
| Simple regression equation has only one coefficient.                                                                                                                 | Multiple regression equation has one coefficient for each independent variable.                                                                                           |
| Equation for simple regression is: $y = \beta_0 + \beta_1x_1$, where $x_1$ is the independent variable.                                                              | Multiple regression equation is represented as: $y = \beta_0 + \beta_1x_1+\beta_2x_2+...+\beta_nx_n$ where $x_i's$ are the independent variables.                         |
| $\beta_0$ is the y-intercept and $\beta_1$ is the slope coefficient, indicating the change in y for a unit change in x.                                              | $\beta_0$ is the y-intercept and $\beta_1, \beta_2, ..., \beta_n$  are the slope coefficients, indicating the impact of each predictor variable on the response variable. |
| Generally, a simple regression cannot predict the target variable effectively as influence on target variable cannot be restricted to a single independent variable. | Multiple regression performs better due to multiple independent predictors, allowing for more precise training and increased accuracy.                                     |
| Simple regression helps understand the direct impact of one variable on other.                                                                                       | Multiple regression helps understand the extent to which two or more explanatory/independent variables and one dependent variable are related.                            |
| Simple regression can be used to predict Electricity bill based on Weather conditions, Height based on Weight, etc.                                                  | Multiple regression can be used to predict the expected yield of a crop at certain levels of rainfall, temperature, and fertilizer addition, etc.                         |
***
#### 37. Explain the concept of feature engineering in Machine Learning.
***
**Ans :**
*Feature Engineering* is the process of extracting and organizing the important features from raw data in such a way that it fits the purpose of the machine learning model. It can be thought of as the art of selecting the important features and transforming them into refined and meaningful features that suit the needs of the model.

***Benefits of Feature Engineering:***
An effective Feature Engineering implies:
- Higher efficiency of the model.
- Easier Algorithms that fit the data.
- Easier for Algorithms to detect patterns in the data.
- Greater Flexibility of the features.
- Better performance of the model and more accurate results.

***Process of Feature Engineering:***
Feature Engineering contains the following processes:

**Feature Transformation:**
1. **Handling Missing Data and Outliers:** Feature engineering deals with inappropriate data, missing values, human interruption, general errors, insufficient data sources, etc. Missing values within the dataset highly affect the performance of the algorithm. A technique called *Imputation* is responsible for handling irregularities within the dataset. Similarly, in this step, outliers are detected using various methods like IQR, Z-score and are substituted with mean, median, mode or other statistical values.
2. **Encoding Categorical Variables**: Categorical variables need to be encoded into numerical representations for machine learning algorithms to process them effectively. Various encoding techniques can be used, including one-hot encoding, label encoding, or target encoding, depending on the nature of the categorical variable and the algorithm being used.
3. **Feature Scaling:** In order to train a predictive model, we need data with a known set of features that needs to be scaled up or down as appropriate. Otherwise, features with high variance and magnitude can introduce bias in the training phase resulting in poor performance. There are multiple methods to scale data, for example, Min-Max scaling where the data is rescaled to the range of 0 and 1. Another method is Standard scaling which shifts the mean of the data to 0 and the standard deviation to 1.

**Feature Construction:**
In this step, new features are constructed from the already present features in the dataset. This helps increase or decrease the size of training data depending on the objective, resulting in more enhanced performance of the model. For example, *Age* can be derived from the feature *Birth Date/Birth Year*, or *Country* can be derived from *country codes* in phone numbers.

**Feature Selection:**
Feature selection aims to identify the most relevant and informative features for the model. Not all features may contribute equally to the predictive power, and including irrelevant features can lead to overfitting or high error. Some of the popular methods for feature selection are: *Forward Selection*, *Backward Elimination*, *Recursive Feature Elimination*, *Information Gain*, etc.

**Feature Extraction:**
Feature extraction involves reducing the number of resources required to describe a large set of data. When performing analysis of complex data one of the major problems stems from the number of variables involved. This is done by employing various algorithms like *Principal Component Analysis*, *Linear Discriminant Analysis*, etc.
***
#### 38. Explain the concept of the confusion matrix in Machine Learning.
***
**Ans :**
*Confusion matrix* is a matrix used to determine the performance of the classification models for a given set of test data. It can only be determined if the true values for test data are known. It shows the correct classification as well as the errors where the classification model incorrectly classifies the variable. Since it shows the errors in the model performance in the form of a matrix, it is also known as an *error matrix*.

***Features of the confusion matrix:***
- For 2 prediction classes, the matrix is a $2 * 2$  table, for 3 classes, it is $3*3$ table, and so on.
- The matrix is divided into two dimensions, that are *predicted values* and *actual values* along with the total number of predictions.
- Predicted values are those values, which are predicted by the model, and actual values are the true values for the given observations.

![[confusion-matrix|400]]

- *True Negative:* The model has given prediction No, and the real or actual value was also No.
- *True Positive:* The model has predicted Yes, and the actual value was also Yes.
- *False Negative:* The model has predicted No, but the actual value was Yes, it is also called as *Type-II error*.
- *False Positive:* The model has predicted Yes, but the actual value was No. It is also called a *Type-I error*.

***Benefit of the confusion matrix:***
- It evaluates the performance of the classification models, when it makes predictions on test data, and tells how good our classification model is.
- It not only tells the error made by the classifiers but also the type of errors such as it is either TYPE-I or TYPE-II error.
- With the help of the confusion matrix, we can calculate the different evaluating metrics for the model, such as accuracy, precision, etc.

***Calculations using confusion matrix:***
1. **Classification Accuracy:** It defines how often the model predicts the correct output. It can be calculated as the ratio of the number of correct predictions made by the classifier to all number of predictions made by the classifiers.
$$Accuracy=\frac{TP+TN}{TP+FP+TN+FN}$$
2. **Precision:** It can be defined as the number of correct outputs provided by the model. In simple words, out of all positive classes that have predicted correctly by the model, how many of them were actually true. It can be calculated using the below formula:
$$Precision=\frac{TP}{TP+FP}$$
3. **Recall:** Recall explains how many of the actual positive cases we were able to predict correctly with our model. Recall for a label is defined as the number of true positives divided by the total number of actual positives. It is also known as *sensitivity*.
$$Recall = \frac{TP}{TP+FN}$$
***
