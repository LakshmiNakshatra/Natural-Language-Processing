# Natural Language Processing

# Importing the dataset
# Strings in the loaded dataset will not be treated as Factors as each word would have different meaning as per the context
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
names(dataset_original)

# Cleaning the text
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)

# Creating volatile corpora consisting of 1000 observations and a column/feature for each word
corpus = VCorpus(VectorSource(dataset_original$Review))
# Converting the text to lower case in order to avoid duplicates columns
corpus = tm_map(corpus, content_transformer(tolower))
# Removing numbers as they are uncorrelated with the results classification 
corpus = tm_map(corpus, removeNumbers)
# Removing punctuations as they are uncorrelated with the results classification 
corpus = tm_map(corpus, removePunctuation)
# Removing non-relevant stopwords words from the corpus. 
corpus = tm_map(corpus, removeWords, stopwords())
# Stemming to obtain the root of each word thus reducing the dimension of feature space 
corpus = tm_map(corpus, stemDocument)
# Removing 'additional' white spaces from corpus.
corpus = tm_map(corpus, stripWhitespace)

# Creating bag of words model
# library(tm)
dtm = DocumentTermMatrix(corpus)
dtm 

# Filter words by having most freq words and removing less freq words
dtm = removeSparseTerms(dtm, 0.999)  

# Creating a dataframe required to fit classification models
dataset = as.data.frame(as.matrix(dtm))

# Adding 'dependent' variable to the dataset
dataset$Liked = dataset_original$Liked

# Encoding the target feature(converting dependent variable as factor)
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(15)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
# library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692], y = training_set$Liked, ntree = 100, mtry=26)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
