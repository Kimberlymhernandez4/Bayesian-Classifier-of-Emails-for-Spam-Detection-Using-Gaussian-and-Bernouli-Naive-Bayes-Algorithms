# Bayesian-Classifier-for-Email-Spam-Using-ML
An ML Bayesian Classifier using the Gaussian and Bernoulli Naive Bayes algorithms. Results will be compared based on their ML performance metrics and accuracy. 
-------------------------------------------------------------------------------------------------------------


# Overview


-------------------------------------------------------------------------------------------------------------

# Problem Domain
Email spam detection is one of the most challenging classification problems in digital communication, and can often contain fraudulent offers, malware links, or other false advertisements. This can directly impact both organizations and individuals as they use online technologies, further raising the need for efficient and accurate filtering methods. Bayesian Classification can be used to evaluate and build a predictive model that can effectively label spam emails. Using UC Irvine’s Spambase dataset, this project aims to determine whether Gaussian Naïve Bayes or Bernoulli Naïve Bayes is more effective in correctly labeling spam and non-spam emails. Bayesian methods are an effective approach to this problem domain because they perform well with limited training data and in cases with a large number of features (Geeks for Geeks, 2025). Additionally, the model is easy to implement and is computationally efficient for predictions and classification analysis. 

----------------------------------------------------------------------------------------------------------

# Objective

The main objective for this analysis is to build a model with Bayesian Classification using the Gaussian Naïve Bayes and Bernoulli Naïve Bayes methods to classify spam and non-spam emails in the Spambase Dataset from the UCI Machine Learning Repository. The main research question is: “Can Bayesian Classification methods effectively label spam and non-spam emails based on text-related features with great accuracy?”. Two models (Gaussian NB and Bernoulli NB) will be compared based on their performance with metrics such as accuracy, recall, precision, f1-score, and the ROC/AUC curve. A secondary question that will be considered is: “ Which Naïve Bayes model is more effective, the Gaussian or Bernoulli algorithm? Based on the performance metrics, why is one model more favorable to use than the other?”

----------------------------------------------------------------------------------------------------------

# Analysis 

-------------------------------------------------------------------------------------------------------------

# Exploratory Analysis
The dataset used for this model is the Spambase dataset, which was collected by researchers and made public in 1999 through the UCI Machine Learning Repository. The dataset contains 4,601 mail entries and 57 columns. Independent variables are comprised of 48-word frequency attributes, 6-character frequency attributes, and 3 attributes that measure capitalization patterns (i.e., the “free” word frequency, the “$” character frequency, and the capitalized running length average). The dependent variable (y label) is the binary classification, which identifies whether the email is spam (1 = True) or non-spam (0 = False). 
The three histograms below provide data visualizations of how selected features are distributed across each class, with blue representing non-spam and orange for spam values: 

<img width="508" height="379" alt="image" src="https://github.com/user-attachments/assets/d1978854-8d07-48d1-8241-6ea33ce5b5c7" />


## **1st Histogram: Word Frequency of “Free”**

Displays the class distribution of ‘word_freq_free’, representing the frequency count of the word “free”. Spam emails are detected with a high count and low frequency, holding some frequency distribution outliers above 2.5. Non-spam emails have a high count of occurrences with a low to near-zero frequency. 

<img width="495" height="370" alt="image" src="https://github.com/user-attachments/assets/7d79c2a9-bdd1-4fc7-bd90-6232757cf460" />

## **2nd Histogram: Character Frequency of “$”**

Displays the class distribution of ‘char_freq_$’, representing the character frequency of ‘$’ (dollar sign character). Spam emails show a higher count at low frequencies and include some minor outliers greater than 1.0. Non-spam emails are near zero in value, with very few occurrences compared to the orange spam group in the histogram. 

<img width="513" height="383" alt="image" src="https://github.com/user-attachments/assets/45eb0766-b5dd-4300-93ab-e367935d2608" />

## **3rd Histogram: Capitalized Running Length Average**
Displays the class distribution of ‘capital_run_length_average’, representing the average length of sequences of consecutive capitalized letters. Non-spam emails are near zero in frequency, which indicates that long sequences of capitalized letters are uncommon. However, in the spam emails, there is a higher frequency that is more notable, with some outliers approaching frequencies close to 100 on the x-axis. 

-------------------------------------------------------------------------------------------------------------


# Preprocessing

For the exploratory analysis, I created a table of key values, which included the rows, columns, spam rate, non-spam count, and spam count of the dataset (displayed below):

<img width="645" height="129" alt="image" src="https://github.com/user-attachments/assets/d92f12f1-4444-42fc-8e45-5a5c18d80226" />
 
The dataset was split into two variables: the target label named ‘class’ (y) and all other columns noted as features (X). Furthermore, the two variables were placed into training and testing sets using the ‘train_test_split’ function (80% train, 20% test) and ‘random_state’ equal to a fixed integer (42). For the Gaussian Naïve Bayes model, features were used in the order provided since the algorithm assumes continuous values and models them as Gaussian distributions. For the Bernoulli Naïve Bayes model, the features were binarized using a threshold to convert frequencies into binary indicators of present (1) or absent (0). Moreover, these preprocessing steps allow the two models to identify whether specific patterns occur at all, regardless of their frequency. 

-------------------------------------------------------------------------------------------------------------

# Model Fitting

Two models (Gaussian Naïve Bayes and Bernoulli Naïve Bayes) were fit using five-fold stratified cross-validation with grid search to identify the best hyperparameters to use. For Gaussian Naïve Bayes, the parameter ‘var_smoothing’, which controls variance regularization, was tuned over values within the Spambase dataset. For Bernoulli Naïve Bayes, a pipeline was created that included a Binarizer, followed by the classifier using GridSearchCV(). Gradients for both algorithms were fit onto the models to create an ROC/AUC curve. The Grid Search tuned the binarization threshold (0.0 to 0.2) and the alpha parameter (0.1 to 5.0) to evaluate the train/test data. 

-------------------------------------------------------------------------------------------------------------


# Model Properties
The Gaussian Naïve Bayes model estimates the variance and mean of each feature for both the spam and non-spam classifiers by applying a smoothing parameter to stabilize small variances. The Bernoulli Naïve Bayes model represents features as binary values (0 and 1) after binarization and alpha tuning with Grid Search for model fitting.  The threshold choice for binarization directly influences how sensitive the models are to small feature frequencies, while the alpha parameter balances the bias-variance tradeoff in probability estimates. 


-------------------------------------------------------------------------------------------------------------

# Output Interpretation

<img width="568" height="139" alt="image" src="https://github.com/user-attachments/assets/7db51185-b6db-48ca-9e94-8350cadfd0fe" />

As shown in the table results above, the Gaussian and Bernoulli methods had varying results. The Gaussian Naïve Bayes achieved an 82.5% accuracy, 70.4% precision score, a recall of 96.1%, an f-1 score of 81.3%, and an ROC/AUC curve of 95.2%. The Bernoulli model had a higher accuracy score (90.2%) and precision (89.3%), but held a significantly lower recall (85.4%). In addition, the Bernoulli model’s f1-score had a greater outcome of 87.3% compared to the Gaussian model, and a slightly improved ROC/AUC curve of 96.0%. Based on the stated objective, we can conclude that the Bernoulli model provided greater quality in results compared to the Gaussian model, which solidifies its reliability in production usage. Overall, the results above can confirm that Bayesian classification is an effective ML approach towards predictive modeling for email spam detection. 


-------------------------------------------------------------------------------------------------------------

# Evaluation

Taking the output interpretation a step further, an ROC/AUC curve was created to assess the performance of both the Gaussian and Bernoulli Naïve Bayes models. This can be viewed in the data visualization below: 


<img width="659" height="492" alt="image" src="https://github.com/user-attachments/assets/32653f30-f9bd-4057-84fb-88160361333a" />

When looking at the ROC curves, both Bayesian models are near 1.0, indicating their high-quality classification between spam and non-spam values. Additionally, the ROC curves show that both models can achieve significantly better results with complex data compared to other classification models, such as logistic and linear regression. When choosing between the Gaussian and Bernoulli models, evaluating all performance metrics is key to ensuring their functionality in detecting binary classifiers. 

-------------------------------------------------------------------------------------------------------------

# Conclusion

-------------------------------------------------------------------------------------------------------------

# Summary

This analysis has demonstrated that Bayesian classification is an effective Machine Learning approach towards detection, using the UCI Spambase dataset. The output results show that both Gaussian Naïve Bayes and Bernoulli Naïve Bayes held high accuracy scores above 82.0% and strong ROC/AUC curves above 95.0%. This meets the stated objective and confirms that the Bayesian Classification approach is a reliable, effective method for spam detection. 

-------------------------------------------------------------------------------------------------------------

# Limitations & Improvement Areas

While the Bernoulli Naïve Bayes model had great performance compared to the Gaussian model, there are some limitations that can be considered. For example, the analysis is limited by the feature engineering choices of the dataset, which includes word and character frequencies, as well as some capitalization metrics. These features do not consider word order and context, which can be supported by other Machine Learning techniques such as TF-IDF vectorization and n-grams. Additionally, Bayesian models can be influenced by irrelevant attributes, which can lead to poor generalization and overfitting when introduced to new, unseen data (Geeks for Geeks, 2025). To improve this, less important features can be removed from the dataset when training the model. Backward Elimination is a great example of this, where the least significant features are removed one at a time based on their contribution to the model’s performance (Cherifa, Asma, 2025). 


-------------------------------------------------------------------------------------------------------------

# References


Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). ‘Spambase [Dataset]. UCI Machine Learning Repository.’ https://doi.org/10.24432/C53G6X.

Geeks For Geeks. (2025). ‘Naïve Bayes Classifiers.’ https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/

Cherifa, Asma. (2025). Medium. ‘Feature selection methods: Backward elimination, forward selection, and LASSO.’ https://medium.com/@asmacherifa/feature-selection-methods-backward-elimination-forward-selection-and-lasso-1b62191a9869

-------------------------------------------------------------------------------------------------------------


