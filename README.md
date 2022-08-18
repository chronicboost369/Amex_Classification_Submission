# Amex_Classification_Submission
Couldn't submit it due to memory limitation to load the test data(33GB)
Training dataset's dimension ~ 5,000,000 x 190

Source of data: https://www.kaggle.com/competitions/amex-default-prediction


# Tasks Performed in the code

1. Data Cleaning: Due to the abudance of samples, some predictors were removed if had missing values were greater than 10% of the total observations. Discovered that
certain observations were missing across several predictors, so identified and removed those.

2. Variable Selection: The original dataset had about 190 variables. Autoencoder & PCA were first used to lower the dimension but didn't yield reliable results.
So, the strategy imposed was selecting variables based on correlations(greater than 0.4 or -0.4) and comparing the averages between the two groups(default and no default). For categorical variables, counts for each level were first used to find somewhat even distribution across all levels. Then, simple logistic regressin was used to check thier efficacy. 

3. Cleaning outliers: Created a function to remove any observatins that are more than 3sd away from the mean. This strategy was chosen because of the abudant samples.

4. Tested Algorithms: Tried to use algorithms that yield probabilities b/c AMEX wanted probability

a. Logistic Regression 2. Singular Vector Machine 3. Random Forest(Best performing) 4. Boosting 5. Gradient Boosting 6. LDA/QDA(didn't pass the multivariable normality assumption).

5. Model Improvedment: Tuned Random Forest with various tree sizes and node sizes. Additionally, tried to find a model with smaller predictors.
The original strategy(selecting variables based on correlation & logstic regression) performed equally well with much smaller predictors.


# Strategies Used to read the large dataset(33GB)

Tried to load them up in chunks, make prediction, delete, and repeat but memory limit was still an issue.
Tried to learn using databases but was taking too long. (*need to learn this)

Tried to use disk.frame but consistently getting:
"Error in fst::write_fst(.SD, file.path(outdir, paste0(.BY, ".fst")), compress = compress) : 
  There was an error during the write operation, fst file might be corrupted. Please check available disk space and access rights.
In addition: Warning messages:"
