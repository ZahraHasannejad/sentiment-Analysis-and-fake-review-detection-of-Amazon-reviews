# Amazon Product Review Analysis

This project analyzes Amazon product reviews to detect fake reviews and perform customer segmentation using machine learning techniques.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Data Preprocessing](#data-preprocessing)
4. [Customer Segmentation](#customer-segmentation)
5. [Fake Review Detection](#fake-review-detection)
6. [Results and Analysis](#results-and-analysis)

## Project Overview

The project aims to:
- Analyze customer behavior through review patterns
- Segment customers based on their reviewing patterns
- Detect potentially fake reviews using unsupervised learning
- Perform sentiment analysis on reviews

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```

## Data Preprocessing

1. Text Processing:
```python
# NLTK preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

2. Custom Functions:
```python
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in stop_words:
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist
```

## Customer Segmentation

1. Feature Engineering:
- Review counts per customer
- Average expenditure
- Positive/negative review ratio
- Review length statistics

2. K-means Clustering:
```python
kmeans = KMeans(n_clusters=6, random_state=47)
clusters = kmeans.fit_predict(dfs1[columns])
```

## Fake Review Detection

1. Text Vectorization:
```python
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9)
text_features = tfidf.fit_transform(dfl['summaryreview_lemma'].values)
```

2. Anomaly Detection:
```python
from sklearn.ensemble import IsolationForest
isolation_forest = IsolationForest(contamination=0.1)
outlier_labels = isolation_forest.fit_predict(outlier_detection_df)
```

## Results and Analysis

### Customer Segments:
- Cluster 0: Moderate reviewers
- Cluster 1: Negative reviewers (potential fake)
- Cluster 2: High-volume reviewers
- Clusters 3-5: Various authentic patterns

### Fake Review Indicators:
1. Extreme sentiment scores
2. Unusual review lengths
3. Irregular voting patterns
4. Suspicious customer behavior

## Conclusions

1. Customer behavior patterns can effectively identify suspicious reviewing activity
2. Combined analysis of text features and numerical metrics improves fake review detection
3. Unsupervised learning techniques successfully segment customers and identify anomalies

## Future Improvements

1. Include more features for analysis
2. Implement supervised learning with labeled data
3. Add real-time detection capabilities
4. Enhance visualization techniques

---
For detailed implementation and code examples, please refer to the Jupyter notebook.

📁 Dataset
The dataset used in this project is not included in this repository.

👉 You can access the original dataset from the following source:

[Ni, J., Li, J., & McAuley, J. (2019, November). Justifying recommendations using distantlylabeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical 
methods in natural language processing and the 9th international joint conference on natural 
language processing (EMNLP-IJCNLP) (pp. 188-197).]

[Amazon Product Data by Julian McAuley in https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/]

If you're using a modified or custom-labeled version of the dataset, please contact the author for more information.

🚨 Notice
⚠️ This repository is part of an ongoing academic research project.
The code is released under the MIT License for educational and non-commercial use.
Please do not reuse this work in publications or derivative projects without proper citation or prior permission.
If you're interested in collaborating, feel free to get in touch!


📬 Contact
Zahra Hasannejad
📧 zahra.hasannejad1998@gmail.com
🌐 GitHub: Zahra Hasannejad

