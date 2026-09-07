from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    X_train,
    X_val,
    X_test,
    max_features=20000
):
    """
    Fit TF-IDF on training data and transform
    validation and test data.
    """
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    return (
        tfidf,
        X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf
    )