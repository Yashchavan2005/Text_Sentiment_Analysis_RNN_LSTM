# ---------------------------------------------------------------
# Project Name : Text_Sentiment_Analysis_RNN_LSTM
# Author       : Yash Chavan
# Date         : 04/10/2026
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Step 1 : Import Required Libraries
# ---------------------------------------------------------------

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------------
# Step 2 : Configuration of Values
# ---------------------------------------------------------------

VOCAB_SIZE = 10000
# Consider the 10,000 most frequently occurring words

MAX_LENGTH = 200
# Consider maximum 200 words from each review


# ---------------------------------------------------------------
# Step 3 : Function to Create Reverse Dictionary
# ---------------------------------------------------------------

def CreateReverseDictionary(word_index):
    """
    Function Name : CreateReverseDictionary

    Description :
        Creates a reverse dictionary to convert word numbers
        back into their corresponding words.

    Input :
        word_index : Word-to-number dictionary

    Output :
        Returns number-to-word dictionary.

    Author : Yash Chavan
    Date   : 04/10/2026
    """

    reverse_word_index = {}

    for word, index in word_index.items():

        # IMDB reserves first 3 indexes for special values
        reverse_word_index[index + 3] = word

    return reverse_word_index


# ---------------------------------------------------------------
# Step 4 : Function to Decode Review
# ---------------------------------------------------------------

def DecodeReview(encoded_review, reverse_word_index):
    """
    Function Name : DecodeReview

    Description :
        Converts an encoded review into readable text.

    Input :
        encoded_review     : Encoded review
        reverse_word_index : Number-to-word dictionary

    Output :
        Returns decoded review as a string.

    Author : Yash Chavan
    Date   : 04/10/2026
    """

    words = []

    for number in encoded_review:

        # Ignore special indexes 0, 1 and 2
        if number >= 3:

            word = reverse_word_index.get(number, "?")

            words.append(word)

    return " ".join(words)


# ---------------------------------------------------------------
# Step 5 : Main Function
# ---------------------------------------------------------------

def main():

    # -----------------------------------------------------------
    # Step 6 : Display Project Information
    # -----------------------------------------------------------

    print("-" * 50)
    print("Text Sentiment Analysis using RNN and LSTM")
    print("-" * 50)

    print("Loading the IMDB dataset...")


    # -----------------------------------------------------------
    # Step 7 : Load IMDB Dataset
    # -----------------------------------------------------------

    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    print("IMDB dataset loaded successfully")

    print("Number of training reviews :", len(X_train))
    print("Number of testing reviews  :", len(X_test))


    # -----------------------------------------------------------
    # Dataset Information
    #
    # X_train -> Reviews used for training
    # Y_train -> Actual sentiment of training reviews
    #
    # X_test  -> Reviews used for testing
    # Y_test  -> Actual sentiment of testing reviews
    #
    # Sentiments :
    # 0 -> Negative
    # 1 -> Positive
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    # Step 8 : Load Word Dictionary
    # -----------------------------------------------------------

    word_index = imdb.get_word_index()

    print("Word dictionary loaded successfully")


    # -----------------------------------------------------------
    # Step 9 : Create Reverse Word Dictionary
    # -----------------------------------------------------------

    reverse_word_index = CreateReverseDictionary(
        word_index
    )

    print("Reverse word dictionary created successfully")


    # -----------------------------------------------------------
    # Step 10 : Display Sample Reviews
    # -----------------------------------------------------------

    print("-" * 50)
    print("Sample Reviews")
    print("-" * 50)

    for i in range(3, 7):

        # Decode encoded review
        review = DecodeReview(
            X_train[i],
            reverse_word_index
        )

        print("-" * 50)

        print("Review Number :", i + 1)

        print("Review :")
        print(review)

        print("-" * 50)

        # Display actual sentiment
        if Y_train[i] == 1:

            print("Sentiment : POSITIVE")

        else:

            print("Sentiment : NEGATIVE")


    # -----------------------------------------------------------
    # Step 11 : Apply Padding
    # -----------------------------------------------------------

    X_train_padded = pad_sequences(
        X_train,
        maxlen=MAX_LENGTH
    )

    X_test_padded = pad_sequences(
        X_test,
        maxlen=MAX_LENGTH
    )

    print("-" * 50)

    print("Training data shape :", X_train_padded.shape)
    print("Testing data shape  :", X_test_padded.shape)

    print("-" * 50)


    # -----------------------------------------------------------
    # Step 12 : Create LSTM Model
    # -----------------------------------------------------------

    model = Sequential()


    # -----------------------------------------------------------
    # Embedding Layer
    # Converts word numbers into 32-dimensional vectors
    # -----------------------------------------------------------

    model.add(
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=32
        )
    )


    # -----------------------------------------------------------
    # LSTM Layer
    # Learns sequence and context of words
    # -----------------------------------------------------------

    model.add(
        LSTM(
            units=64
        )
    )


    # -----------------------------------------------------------
    # Dense Output Layer
    # Produces probability between 0 and 1
    # -----------------------------------------------------------

    model.add(
        Dense(
            units=1,
            activation="sigmoid"
        )
    )


    # -----------------------------------------------------------
    # Model Architecture
    #
    # Review
    #    ↓
    # Embedding
    #    ↓
    # LSTM
    #    ↓
    # Dense
    #    ↓
    # Sigmoid
    #    ↓
    # Positive / Negative
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    # Step 13 : Compile the Model
    # -----------------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Model compiled successfully")


    # -----------------------------------------------------------
    # Step 14 : Train the Model
    # -----------------------------------------------------------

    print("-" * 50)
    print("Model Training Started")
    print("-" * 50)

    model.fit(
        X_train_padded,
        Y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2
    )

    print("-" * 50)
    print("Model training completed")
    print("-" * 50)


    # -----------------------------------------------------------
    # Step 15 : Evaluate the Model
    # -----------------------------------------------------------

    loss, accuracy = model.evaluate(
        X_test_padded,
        Y_test,
        verbose=0
    )

    print("Testing Loss     :", loss)
    print("Testing Accuracy :", accuracy)


    # -----------------------------------------------------------
    # Step 16 : Select Test Review
    # -----------------------------------------------------------

    TEST_REVIEW_NUMBER = 0

    original_review = X_test[
        TEST_REVIEW_NUMBER
    ]

    # Decode review into readable text
    decoded_review = DecodeReview(
        original_review,
        reverse_word_index
    )

    print("-" * 50)
    print("Review Given to the Model")
    print("-" * 50)

    print(decoded_review)


    # -----------------------------------------------------------
    # Step 17 : Get Actual Sentiment
    # -----------------------------------------------------------

    actual_value = Y_test[
        TEST_REVIEW_NUMBER
    ]

    if actual_value == 1:

        actual_sentiment = "POSITIVE"

    else:

        actual_sentiment = "NEGATIVE"

    print("-" * 50)
    print("Actual Sentiment :", actual_sentiment)


    # -----------------------------------------------------------
    # Step 18 : Predict Sentiment
    # -----------------------------------------------------------

    review_for_prediction = X_test_padded[
        TEST_REVIEW_NUMBER:
        TEST_REVIEW_NUMBER + 1
    ]

    prediction = model.predict(
        review_for_prediction,
        verbose=0
    )

    # Extract prediction probability
    probability = prediction[0][0]


    # -----------------------------------------------------------
    # Step 19 : Convert Probability into Sentiment
    # -----------------------------------------------------------

    if probability >= 0.5:

        predicted_sentiment = "POSITIVE"

    else:

        predicted_sentiment = "NEGATIVE"


    # -----------------------------------------------------------
    # Step 20 : Display Final Result
    # -----------------------------------------------------------

    print("-" * 50)
    print("Final Result")
    print("-" * 50)

    print("Prediction Probability :", probability)
    print("Actual Sentiment       :", actual_sentiment)
    print("Predicted Sentiment    :", predicted_sentiment)

    print("-" * 50)


# ---------------------------------------------------------------
# Step 21 : Main Function Calling
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()
