## ADDED Requirements

### Requirement: Sentiment dataset loading
The system SHALL load the sentiment dataset from `data/sentiment_data.csv` using pandas and validate that it contains columns `ID`, `text`, and `emotion`.

#### Scenario: Dataset schema is validated
- **WHEN** the training pipeline is executed
- **THEN** the dataset is loaded from `data/sentiment_data.csv`
- **AND** the required columns (`ID`, `text`, `emotion`) are present

### Requirement: Text preprocessing and tokenization
The system SHALL clean and normalize input text, tokenize it, and remove stopwords using NLTK (or equivalent deterministic stopword lists).

#### Scenario: Text is transformed into normalized tokens
- **WHEN** a raw text row is provided to the preprocessing pipeline
- **THEN** the output is a cleaned representation suitable for vectorization
- **AND** stopwords are removed

### Requirement: TF-IDF feature extraction
The system SHALL convert preprocessed text into TF-IDF vectors with `max_features = 10000`.

#### Scenario: Vectorizer configuration is enforced
- **WHEN** the model training pipeline is run
- **THEN** TF-IDF features are produced
- **AND** the configured feature cap is `max_features = 10000`

### Requirement: Logistic Regression training and split
The system SHALL train a scikit-learn Logistic Regression classifier using an 80/20 train/test split.

#### Scenario: Model is trained and evaluated on held-out data
- **WHEN** the training pipeline is executed
- **THEN** 80% of the dataset is used for training and 20% for testing
- **AND** a Logistic Regression model is fit on the training set
- **AND** predictions are generated for the test set

### Requirement: Evaluation outputs
The system SHALL print model accuracy, a classification report, and a confusion matrix for the test set.

#### Scenario: Metrics are produced after training
- **WHEN** training completes
- **THEN** accuracy is printed
- **AND** a classification report is printed
- **AND** a confusion matrix is printed

### Requirement: Model persistence
The system SHALL save the trained sentiment model and vectorizer to disk using pickle and SHALL support loading them for gameplay inference.

#### Scenario: Saved model is reused for inference
- **WHEN** a saved pickle artifact is present
- **THEN** the game runtime loads the persisted model/vectorizer
- **AND** sentiment predictions can be produced without retraining

### Requirement: Runtime sentiment prediction labels
The system SHALL map sentiment inference outputs into three runtime labels: `positive`, `neutral`, and `negative`.

#### Scenario: Feedback text is classified into a runtime sentiment label
- **WHEN** the player provides post-game feedback text
- **THEN** the system predicts one of `positive`, `neutral`, or `negative`
