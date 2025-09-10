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

















