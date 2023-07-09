import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

articles_df = pd.DataFrame(articles_list)


class NYTClassifier:
    def __init__(self, categorical_variable, category_values, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.categorical_variable = categorical_variable
        self.category_values = category_values
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
    
        for _ in range(self.n_iter):
            errors = 0
    
            for _, row in X.iterrows():
                xi = row.values
                target = y[row.name]
    
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        return self

    def preprocessing(self, articles_df, class1, class2):
        articles_df["pub_date"] = pd.to_datetime(articles_df["pub_date"])
        
        if self.categorical_variable == "half_year":
            half_year1 = pd.to_datetime("2022-01-01 00:00:00").tz_localize('UTC')
            half_year2 = pd.to_datetime("2022-07-01 00:00:00").tz_localize('UTC')
            X = articles_df.copy()
                
            y = pd.Series(-1, index=X.index)  # Assign -1 to all rows by default
            y[(X["pub_date"] >= half_year1) & (X["pub_date"] < half_year2)] = 1  # Assign 1 to the rows within the specified half-year range

            
        elif self.categorical_variable == "section_name":
            X = articles_df[articles_df[self.categorical_variable].isin([class1, class2])].copy()
            y = pd.Series(1, index=X.index)
            y[~X[self.categorical_variable].str.contains(class2)] = -1
        else:
            raise ValueError("Invalid categorical_variable value.")
        
        X["sentence_length"] = X["lead_paragraph"].apply(lambda x: len(nltk.sent_tokenize(x)))
        X["word_length"] = X["lead_paragraph"].apply(lambda x: len(nltk.word_tokenize(x)))
        X["sentence_count"] = X["lead_paragraph"].apply(lambda x: len(nltk.sent_tokenize(x)))
        X["noun_ratio"] = X["lead_paragraph"].apply(lambda x: sum(1 for token in nltk.pos_tag(nltk.word_tokenize(x)) if token[1].startswith("NN")) / len(nltk.word_tokenize(x)) if len(nltk.word_tokenize(x)) > 0 else 0)
        X["verb_ratio"] = X["lead_paragraph"].apply(lambda x: sum(1 for token in nltk.pos_tag(nltk.word_tokenize(x)) if token[1].startswith("VB")) / len(nltk.word_tokenize(x)) if len(nltk.word_tokenize(x)) > 0 else 0)
        X["stopword_ratio"] = X["lead_paragraph"].apply(lambda x: sum(1 for token in nltk.word_tokenize(x) if token.lower() in nltk.corpus.stopwords.words("english")) / len(nltk.word_tokenize(x)) if len(nltk.word_tokenize(x)) > 0 else 0)
        
        X = X[["sentence_length", "word_length", "sentence_count", "noun_ratio", "verb_ratio", "stopword_ratio"]]
        
        return X, y

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)






######## Half year category ############

classifier1 = NYTClassifier(categorical_variable="half_year", category_values=["half_year1", "half_year2"])
train_test1 = classifier1.preprocessing(articles_df, "half_year1","half_year2")


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(train_test1[0], train_test1[1], test_size=0.2, random_state=1)

# Fit the classifier on the training data
classifier1.fit(X_train, y_train)

y_train_pred = classifier1.predict(X_train)
y_test_pred = classifier1.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

errors = classifier1.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title("Errors")
plt.show()

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)



#iterating over the half_year with different learning rates is possible, but requires a lot of time 
#since the size of the dataframe is too large

######### section_name category #########


############################################################
############################################################


#iterating doesn't seem to very effective, Size of the operation is too large
#doing everything step by step
#Case 1 Sports and Style

learning_rate = 0.01
classifier = NYTClassifier(categorical_variable="section_name", category_values=["Sports", "Style"])

X, y = classifier.preprocessing(articles_df, "Sports", "Style")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier.eta = learning_rate  # Set the learning rate of the classifier
classifier.fit(X_train, y_train)

errors = classifier.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title(f'Learning Rate: {learning_rate}')
plt.show()

#other learning rate
learning_rate = 0.25
classifier.eta = learning_rate  # Set the learning rate of the classifier
classifier.fit(X_train, y_train)

errors = classifier.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title(f'Learning Rate: {learning_rate}')
plt.show()


#Case 2 Sports and Books

learning_rate = 0.01
classifier = NYTClassifier(categorical_variable="section_name", category_values=["Sports", "Books"])

X, y = classifier.preprocessing(articles_df, "Sports", "Books")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier.eta = learning_rate  # Set the learning rate of the classifier
classifier.fit(X_train, y_train)

errors = classifier.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title(f'Learning Rate: {learning_rate}')
plt.show()

#other learning rate
learning_rate = 0.5
classifier.eta = learning_rate  # Set the learning rate of the classifier
classifier.fit(X_train, y_train)

errors = classifier.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title(f'Learning Rate: {learning_rate}')
plt.show()


#Case 3 Style and Books

learning_rate = 0.01
classifier = NYTClassifier(categorical_variable="section_name", category_values=["Style", "Books"])

X, y = classifier.preprocessing(articles_df, "Style", "Books")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier.eta = learning_rate  # Set the learning rate of the classifier
classifier.fit(X_train, y_train)

errors = classifier.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title(f'Learning Rate: {learning_rate}')
plt.show()

#other learning rate
learning_rate = 0.1
classifier.eta = learning_rate  # Set the learning rate of the classifier
classifier.fit(X_train, y_train)

errors = classifier.errors_
epoch = np.arange(1, len(errors) + 1)
error_percentages = np.array(errors) / len(y_train) * 100

plt.plot(epoch, error_percentages)
plt.xlabel('Epoch')
plt.ylabel('Error Percentage')
plt.title(f'Learning Rate: {learning_rate}')
plt.show()







