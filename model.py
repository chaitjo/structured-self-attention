from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.regularizers import *
from keras.initializers import *


def build_structured_self_attention_embedder(word_window_size, vocabulary_size, word_embedding_size,
                                             hidden_state_size, num_layers, attention_filters1, attention_filters2,
                                             dropout, recurrent_dropout, regularization_lambda):
    # Input for text sequence
    sequence_input = Input(shape=(word_window_size,), name="sequence_input_placeholder")

    # Word embeddings lookup for words in sequence
    sequence_word_embeddings = Embedding(input_dim=vocabulary_size + 1,
                                         output_dim=word_embedding_size,
                                         embeddings_initializer='glorot_uniform',
                                         embeddings_regularizer=l2(regularization_lambda),
                                         mask_zero=True,
                                         name="sequence_word_embeddings")(sequence_input)

    # Obtain hidden state of Bidirectional LSTM at each word embedding
    hidden_states = sequence_word_embeddings
    for layer in range(num_layers):
        hidden_states = Bidirectional(LSTM(units=hidden_state_size,
                                           dropout=dropout,
                                           recurrent_dropout=recurrent_dropout,
                                           kernel_initializer='glorot_uniform',
                                           recurrent_initializer='glorot_uniform',
                                           bias_initializer='zeros',
                                           kernel_regularizer=l2(regularization_lambda),
                                           recurrent_regularizer=l2(regularization_lambda),
                                           bias_regularizer=l2(regularization_lambda),
                                           activity_regularizer=l2(regularization_lambda),
                                           implementation=1,
                                           return_sequences=True,
                                           return_state=False,
                                           unroll=True),
                                      merge_mode='concat', name="lstm_outputs_{}".format(layer))(hidden_states)

    # Attention mechanism
    attention = Conv1D(filters=attention_filters1, kernel_size=1, activation='tanh', padding='same', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros',
                       kernel_regularizer=l2(regularization_lambda),
                       bias_regularizer=l2(regularization_lambda), activity_regularizer=l2(regularization_lambda),
                       name="attention_layer1")(hidden_states)
    attention = Conv1D(filters=attention_filters2, kernel_size=1, activation='linear', padding='same', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros',
                       kernel_regularizer=l2(regularization_lambda),
                       bias_regularizer=l2(regularization_lambda), activity_regularizer=l2(regularization_lambda),
                       name="attention_layer2")(attention)
    attention = Lambda(lambda x: softmax(x, axis=1), name="attention_vector")(attention)

    # Apply attention weights
    weighted_sequence_embedding = Dot(axes=[1, 1], normalize=False, name="weighted_sequence_embedding")(
        [attention, hidden_states])

    # Add and normalize to obtain final sequence embedding
    sequence_embedding = Lambda(lambda x: K.l2_normalize(K.sum(x, axis=1)))(weighted_sequence_embedding)

    # Build model
    model = Model(inputs=sequence_input, outputs=sequence_embedding, name="sequence_embedder")
    model.summary()
    return model
