# Reddit_Classification_Pyspark

## 1. Overall Project Goal

You set out to **classify Reddit posts into their respective subreddits** using both traditional machine learning (via scikit-learn) and distributed computing (via PySpark). The motivation behind this was to manage the vast amount of content on Reddit and automatically categorize posts, thereby improving content discoverability and user engagement.

---

## 2. Methodology

1. **Environment Setup**
    - Installed and configured necessary packages (e.g., PySpark, scikit-learn, nltk).
    - Set up the environment to handle large datasets, ensuring sufficient computational resources.
2. **Data Extraction**
    - Downloaded raw Reddit data from Pushshift.io for multiple subreddits.
    - Focused on five subreddits: personalfinance, travel, cooking, plants, and programming.
3. **Data Conversion**
    - Converted the downloaded `.zst_blocks` format to more usable `.zst_files`.
    - Further transformed these into CSV files, merging them into a single dataset.
4. **Data Ingestion**
    - Loaded the CSV files into both Python (pandas/scikit-learn) and Spark DataFrames for subsequent processing.
5. **Data Cleaning**
    - Removed irrelevant or redundant columns (like URLs and ‘permalink’).
    - Dropped rows with missing or deleted text (e.g., “deleted” or “removed” selftext).
    - Dealt with NaN values and standardized Boolean/categorical fields.
6. **Exploratory Data Analysis (EDA)**
    - Investigated distributions for numeric columns (e.g., num_comments, score).
    - Visualized categorical data (e.g., which subreddits had the most posts).
    - Performed text analysis on the ‘title’ and ‘selftext’ columns to identify common words, bigrams, and trigrams.
7. **Feature Engineering**
    - Transformed text fields into numerical features using NLP techniques (TF-IDF, HashingTF, IDF, etc.).
    - Scaled numeric columns to ensure consistent ranges.
    - Encoded categorical variables (e.g., link_flair_text) using ordinal or string indexing.
8. **Model Training**
    - Implemented Random Forest, Decision Tree, and Logistic Regression in both scikit-learn and PySpark.
    - Tuned hyperparameters (e.g., number of estimators, max_depth) via GridSearchCV in scikit-learn.
    - Adapted the best-performing hyperparameters to PySpark’s MLlib.
9. **Model Evaluation**
    - Assessed models using accuracy, precision, recall, and F1-score on validation and test datasets.
    - Compared performance across scikit-learn and PySpark to highlight differences in distributed vs. single-node approaches.
10. **Visualization**
- Presented confusion matrices and performance metrics for each model.
- Showcased differences between scikit-learn and PySpark results.
    
    ![image.png](attachment:90406289-5331-42b7-b59c-08970eb98c93:image.png)
    

---

## 3. Data Transformation and Pipeline Building (scikit-learn)

In your scikit-learn workflow, you used a multi-step pipeline to process data before training:

1. **Text Processing**
    - Used **nltk** to remove stopwords and apply lemmatization.
    - Cleaned and tokenized both the ‘title’ and ‘selftext’ columns.
2. **Data Encoding**
    - **LabelEncoder**: Converted your target variable (subreddit) from string labels to numeric codes.
    - **OrdinalEncoder** (or similar encoders) for columns like link_flair_text.
3. **Vectorization**
    - **TF-IDF**: Transformed tokenized text into numerical vectors that capture term importance.
    - This step created high-dimensional vectors for each text field.
4. **Standardization**
    - **StandardScaler**: Scaled numeric columns (e.g., score, num_comments) to have zero mean and unit variance.
5. **Dropping Unused Columns**
    - Retained only columns relevant to the model (e.g., text-based features, numeric features, link_flair_text).
    - Dropped any columns deemed non-informative or highly correlated duplicates.

The final output of the pipeline was a fully transformed feature set ready for model input.

![image.png](attachment:53833616-f580-4d8e-9966-ead2ae66eff5:image.png)

---

## 4. PySpark Workflow: Transformers and Pipeline Stages

When using **PySpark**, you replicated many of the same transformations but leveraged Spark’s distributed processing. Your PySpark pipeline included:

1. **Target Label Encoding**
    - Used **StringIndexer** to convert subreddit labels into numeric indices (with `handleInvalid="keep"` to handle unexpected labels).
2. **Categorical Columns**
    - Additional **StringIndexer** transformations for other categorical columns (like link_flair_text), again with `handleInvalid="keep"`.
3. **Text Columns Stages**
    - **RegexTokenizer**: Broke text into tokens.
    - **StopWordsRemover**: Removed common words (the same idea as nltk stopword removal).
    - **HashingTF**: Transformed tokens into term-frequency vectors.
    - **IDF**: Calculated inverse document frequency, weighting less-common terms more heavily.
4. **Numeric Columns**
    - Combined numeric fields using **VectorAssembler**.
    - Applied **StandardScaler** to ensure consistent feature scales.
5. **Combined Features Vector**
    - Used **VectorAssembler** to merge text-based TF-IDF features, numeric columns, and categorical encodings into a single vector for the model.
6. **Model Training**
    - Configured a chosen classifier (e.g., Random Forest) with Spark MLlib’s parameters (like `numTrees=400`, `maxBins=40`, etc.).
    - Fit the pipeline on training data, then applied the same pipeline to validation and test data.

---

## 5. Model Evaluation (PySpark)

You evaluated three main models in PySpark using **MulticlassClassificationEvaluator**:

1. **Random Forest**
    - Validation Accuracy: ~0.89
    - Test Accuracy: ~0.89
    - F1 Score: ~0.8782
2. **Decision Tree**
    - Validation Accuracy: ~0.99
    - Test Accuracy: ~0.99
    - F1 Score: ~0.9932
3. **Logistic Regression**
    - Validation Accuracy: ~0.68
    - Test Accuracy: ~0.67
    - F1 Score: ~0.6306

The Decision Tree achieved exceptionally high accuracy (99%), while the Random Forest had strong overall metrics. Logistic Regression lagged in comparison, which is not uncommon for highly imbalanced text classification tasks without extensive regularization or additional data augmentation.

---

![image.png](attachment:11a5bf94-8c65-4ecf-86d1-3803974d996c:image.png)

## 6. Model Evaluation: scikit-learn vs. PySpark

You noted several key differences between scikit-learn and PySpark evaluations:

- **Implementation Variances**
Different algorithmic optimizations can lead to slightly different results.
- **Data Handling**
    - PySpark distributes data across a cluster.
    - scikit-learn processes data on a single machine.
- **Default Settings**
    - scikit-learn and PySpark may have different default parameter values (e.g., max iterations, regularization).
- **Preprocessing**
    - Variation in how data is normalized or scaled can shift model performance.
- **Random Processes**
    - Each library can differ in random seeds, shuffling strategies, and parallelization, affecting reproducibility.

Despite these nuances, the Random Forest model was consistently a top performer in both environments, underscoring its robustness.

---

## 7. Additional Observations and Insights

1. **Dropping the ‘programming’ Subreddit**
    - This subreddit had very few data points (~147 rows), making it too small for robust modeling. Removing it helped prevent skewed results.
2. **Handling Imbalanced Data**
    - Subreddits like personalfinance and travel had more posts than others, potentially skewing the classifier. You used stratified splits to mitigate this.
3. **Text-Only vs. Combined Features**
    - You also experimented with models using only text features (title and selftext) versus including numeric/categorical metadata.
    - Random Forest again emerged as a strong candidate, often achieving accuracy above 90%.
4. **Future Directions**
    - **Real-Time Classification**: Implement a streaming approach with Spark Structured Streaming.
    - **Advanced NLP**: Consider more sophisticated embeddings (e.g., Word2Vec, BERT) to capture deeper language context.
    - **Fine-Grained Categories**: Extend the model to classify among more subreddits or detect multi-label scenarios if a post spans multiple themes.
        
        ![image.png](attachment:cf141bc9-0513-4157-90a0-183601f67951:image.png)
        

---

## 8. Conclusion

Your project showcased the **end-to-end application of data science and big data engineering** to a real-world classification problem:

- You **extracted and cleaned** a large Reddit dataset.
- Performed **EDA** to understand numeric and text distributions.
- Built a **multi-step pipeline** in both scikit-learn and PySpark to transform raw text, numeric, and categorical data into a unified feature set.
- Trained and evaluated **multiple models** (Random Forest, Decision Tree, Logistic Regression), achieving **high accuracy** (up to ~99% in some scenarios).
- Demonstrated how **PySpark** can scale machine learning tasks for larger datasets while scikit-learn remains a simpler, more direct approach for smaller or moderately sized data.

By successfully deploying these pipelines, you provided a solid foundation for more advanced solutions, such as real-time classification or more nuanced NLP-driven feature extraction. Ultimately, your work illustrates how **big data technologies** (like Apache Spark) can meaningfully improve the efficiency and scalability of machine learning workflows, especially when dealing with **large-scale text data** such as Reddit posts.

***Note: Used Chat Gpt to articulate***
