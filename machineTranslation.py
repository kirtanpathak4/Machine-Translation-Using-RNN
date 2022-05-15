# 1.	Kirtan Sandeep Pathak – KSP200004
# 2.	Krishna Vinaybhai Patel – KVP200006
# 3.	Parth Hareshbhai Sabhadiya – PXS210002
# 4.	Ankit Upadhyay – AXU200010


import os
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pandas import read_excel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wget


def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_seq(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


class DataPipe:
    def __init__(self, df_array, clip_vocab_size=500, test_size=0.15):
        assert isinstance(df_array, np.ndarray) and df_array.shape[
            1] == 2, "Expected 2d Numpy array with only two column original_lang, target_lang"

        self.independent_token = tokenization(df_array[:, 0])
        self.independent_token_size = len(
            self.independent_token.word_counts) + 1
        self.target_token = tokenization(df_array[:, 1])
        self.target_token_size = len(self.target_token.word_counts) + 1
        train, val = train_test_split(
            df_array, test_size=test_size, random_state=42)
        self.token_size = clip_vocab_size
        self.x_train, self.y_train = encode_seq(self.independent_token, clip_vocab_size, train[:, 0]), encode_seq(
            self.target_token, clip_vocab_size, train[:, 1])
        self.x_val, self.y_val = encode_seq(self.independent_token, clip_vocab_size, val[:, 0]), encode_seq(
            self.target_token, clip_vocab_size, val[:, 1])
        self.train_seeker = 0
        self.validation_seeker = 0

    def fetch(self, df_x, df_y):
        for x, y in zip(df_x, df_y):
            yield x, y

    def fetch_train(self, range=None):
        if range is None:
            range = self.x_train.shape[0]
        train_seeker = self.train_seeker
        self.train_seeker = (self.train_seeker +
                             range) % (self.x_train.shape[0])
        return self.x_train[train_seeker: train_seeker + range], self.y_train[train_seeker: train_seeker + range]

    def fetch_val(self, range=None):
        if range is None:
            range = self.x_val.shape[0]
        val_seeker = self.validation_seeker
        self.validation_seeker = (
            self.validation_seeker + range) % (self.x_val.shape[0])
        return self.x_val[val_seeker: val_seeker + range], self.y_val[val_seeker: val_seeker + range]


class MyRnnModel:

    def __init__(self, input_layer, hidden_layer, output_layer, sentance_length, learning_rate, data_object):
        self.hidden_layer = hidden_layer
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.sentance_length = sentance_length
        self.learning_rate = learning_rate
        self.input_to_hidden_weight = np.random.randn(
            self.input_layer, self.hidden_layer)

        self.default_hidden_state = np.zeros((1, self.hidden_layer))
        self.hidden_to_hidden_weight = np.random.randn(
            self.hidden_layer, self.hidden_layer)
        self.hidden_bias = np.zeros((1, self.hidden_layer))

        self.hidden_to_output_weight = np.random.randn(
            self.hidden_layer, self.output_layer)
        self.output_bias = np.zeros((1, self.output_layer))

        self.memoryInput = np.zeros_like(self.input_to_hidden_weight)
        self.memoryHidden = np.zeros_like(self.hidden_to_hidden_weight)
        self.memoryOutput = np.zeros_like(self.hidden_to_output_weight)
        self.memoryHiddenBias = np.zeros_like(self.hidden_bias)
        self.memoryOutputBias = np.zeros_like(self.output_bias)

        self.data_object = data_object
        self.target_word_lookup = {
            val: key for key, val in data_object.target_token.word_index.items()}
        self.independent_word_lookup = {
            val: key for key, val in data_object.independent_token.word_index.items()}
        self.train_loss_track = []
        self.val_loss_track = []

    def describe_my_network(self):
        return (f"""
              (
                (
                  (
                    ({self.input_to_hidden_weight.shape} X {self.hidden_to_hidden_weight.shape}) + {self.hidden_bias.shape}
                  ) X {self.hidden_to_output_weight.shape}
                ) + {self.output_bias.shape}
              )
        """.strip(""))

    def softmax(self, x):
        p = np.exp(x)
        return np.divide(p, np.sum(p, axis=1)[:, None])

    def forward_propogation(self, encoded_input_sentance):
        ip_tracker, hidden_tracker, output_tracker, probability = {}, {}, {}, {}
        hidden_tracker[-1] = np.zeros(
            (encoded_input_sentance.shape[0], self.hidden_layer))
        for t in range(encoded_input_sentance.shape[1]):
            temp = encoded_input_sentance[:, t]
            ip_tracker[t] = np.zeros((len(temp), self.input_layer))
            for idx in tuple(zip(range(len(temp)), temp.ravel())):
                ip_tracker[t][idx] = 1
            hidden_tracker[t] = np.tanh((ip_tracker[t] @ self.input_to_hidden_weight) + (
                hidden_tracker[t - 1] @ self.hidden_to_hidden_weight) + self.hidden_bias)
            output_tracker[t] = hidden_tracker[t] @ self.hidden_to_output_weight + \
                self.output_bias
            probability[t] = self.softmax(output_tracker[t])
        return ip_tracker, hidden_tracker, probability

    def backward_propogation(self, input_states, hidden_states, predicted_probability, y_train):
        delta_inputW, delta_hiddenW, delta_outputW = np.zeros_like(self.input_to_hidden_weight), np.zeros_like(
            self.hidden_to_hidden_weight), np.zeros_like(self.hidden_to_output_weight)
        delta_hiddenB, delta_outputB = np.zeros_like(
            self.hidden_bias), np.zeros_like(self.output_bias)
        delta_hidden_new = np.zeros_like((hidden_states[0]))
        for t in range(self.sentance_length - 1, -1, -1):
            delta_y = np.copy(predicted_probability[t])
            target_state = y_train[:, t]
            for i, val in enumerate(target_state):
                delta_y[(i, val)] -= 1
            delta_outputW = delta_outputW + hidden_states[t].T @ delta_y
            delta_outputB = delta_outputB+delta_outputB
            dh = delta_y @ self.hidden_to_output_weight.T + delta_hidden_new
            dhrec = (1 - hidden_states[t] * hidden_states[t]) * dh

            delta_hiddenB = delta_hiddenB+dhrec
            delta_inputW = delta_inputW+np.dot(input_states[t].T, dhrec)
            delta_hiddenW = delta_hiddenW+np.dot(hidden_states[t].T, dhrec)

            delta_hidden_new = dhrec @ self.hidden_to_hidden_weight

        delta_inputW = np.clip(delta_inputW, -1, 1)
        self.memoryInput = self.memoryInput+delta_inputW * delta_inputW
        self.input_to_hidden_weight = self.input_to_hidden_weight - \
            self.learning_rate * delta_inputW / \
            np.sqrt(self.memoryInput + 1e-8)

        delta_hiddenW = np.clip(delta_hiddenW, -1, 1)
        self.memoryHidden = self.memoryHidden+delta_hiddenW * delta_hiddenW
        self.hidden_to_hidden_weight = self.hidden_to_hidden_weight - \
            self.learning_rate * delta_hiddenW / \
            np.sqrt(self.memoryHidden + 1e-8)

        delta_outputW = np.clip(delta_outputW, -1, 1)
        self.memoryOutput = self.memoryOutput+delta_outputW * delta_outputW
        self.hidden_to_output_weight = self.hidden_to_output_weight - \
            self.learning_rate * delta_outputW / \
            np.sqrt(self.memoryOutput + 1e-8)

        delta_hiddenB = np.clip(delta_hiddenB.mean(
            axis=0), -0.5, 0.5).reshape(1, -1)
        self.memoryHiddenBias = self.memoryHiddenBias+delta_hiddenB * delta_hiddenB
        self.hidden_bias = self.hidden_bias-self.learning_rate * \
            delta_hiddenB / np.sqrt(self.memoryHiddenBias + 1e-8)

        delta_outputB = np.clip(delta_outputB.mean(
            axis=0), -0.5, 0.5).reshape(1, -1)
        self.memoryOutputBias = self.memoryOutputBias+delta_outputB * delta_outputB
        self.output_bias = self.output_bias-self.learning_rate * \
            delta_outputB / np.sqrt(self.memoryOutputBias + 1e-8)

    def get_loss(self, predicted, target):
        return np.sqrt(np.mean(-np.log([predicted[t][idx, val] for t in range(self.sentance_length) for idx, val in enumerate(target[:, t])])))

    def say_the_sentance(self, probability):

        return " ".join(
            [self.target_word_lookup.get(np.argmax(probability[word]), "") for word in range(self.sentance_length)])

    def train(self, n_epochs, training_size=None):
        for _ in tqdm(range(n_epochs)):
            x_train, y_train = self.data_object.fetch_train(
                range=training_size)
            ip_tracker, hidden_tracker, probability = self.forward_propogation(
                encoded_input_sentance=x_train)
            loss = self.get_loss(predicted=probability, target=y_train)
            self.backward_propogation(input_states=ip_tracker, hidden_states=hidden_tracker,
                                      predicted_probability=probability, y_train=y_train)
            self.train_loss_track.append(loss)
            self.val_loss_track.append(self.calculate_validation_loss())

    def calculate_validation_loss(self):
        x_val, y_val = self.data_object.fetch_val(range=None)
        _, _, probability = self.forward_propogation(
            encoded_input_sentance=x_val)
        return self.get_loss(predicted=probability, target=y_val)

    def translate(self, input_sentance, word_length):
        encoded_sentance = encode_seq(
            self.data_object.independent_token, word_length, [input_sentance])
        _, _, probability = self.forward_propogation(
            encoded_input_sentance=encoded_sentance)
        return self.say_the_sentance(probability=probability)


class MachineTransalationPipeLine:

    def __init__(self, translate_from, translate_to, sentance_length):
        self.translate_from = translate_from
        self.translate_to = translate_to
        self.sentance_length = sentance_length
        self.data_manager = None
        self.model = None

    @staticmethod
    def read_and_massage_data(file_path, nrows=1e5):
        def massage_data(s): return s.translate(
            str.maketrans('', '', string.punctuation)).lower()
        df = read_excel(file_path, usecols=[
                        0, 1], nrows=nrows, engine='openpyxl').applymap(massage_data)
        return df

    def make_data_pipeline(self, data, test_size=0.2):
        data = data[[self.translate_from, self.translate_to]]
        self.data_manager = DataPipe(df_array=data.to_numpy(
        ), clip_vocab_size=self.sentance_length, test_size=test_size)
        print("Training Size : {}, Validation Size : {}".format(self.data_manager.x_train.shape,
                                                                self.data_manager.x_val.shape))

    def initialize_model(self, hidden_nodes, learning_rate):
        self.model = MyRnnModel(input_layer=self.data_manager.independent_token_size, hidden_layer=hidden_nodes,
                                output_layer=self.data_manager.target_token_size,
                                sentance_length=self.data_manager.token_size, learning_rate=learning_rate,
                                data_object=self.data_manager)

    def train_model(self, n_epochs, training_size=None, validation_size=5, validate_every=50):
        self.model.train(n_epochs=n_epochs, training_size=training_size)

    def translate_from_model(self, statement, word_length):
        return self.model.translate(statement, word_length)

    def execute_training_pipeline(self, data_path, nrows, hidden_nodes, learning_rate,
                                  n_epochs, training_size=None, test_size=0.2):
        data = MachineTransalationPipeLine.read_and_massage_data(
            file_path=data_path, nrows=nrows)
        self.make_data_pipeline(data=data, test_size=test_size)
        self.initialize_model(hidden_nodes=hidden_nodes,
                              learning_rate=learning_rate)
        self.train_model(n_epochs=n_epochs, training_size=training_size)
        print(np.mean(self.model.train_loss_track))
        # self.model.calculate_accuracy()
        print(np.mean(self.model.val_loss_track))

    def execute_inference_pipeline(self, statement):
        return self.translate_from_model(statement=statement, word_length=self.sentance_length)


def train_and_infer(nrows, learning_rate, epochs, hidden_nodes, training_size, counter, sentance_length, fileName):
    translator = MachineTransalationPipeLine(translate_from='English', translate_to='French',
                                             sentance_length=sentance_length)
    translator.execute_training_pipeline(data_path=fileName,
                                         nrows=nrows, hidden_nodes=hidden_nodes,
                                         learning_rate=learning_rate,
                                         n_epochs=epochs, training_size=training_size, test_size=0.2)
    plt.plot(range(len(translator.model.val_loss_track)), translator.model.val_loss_track,
             label="Val Loss")
    plt.plot(range(len(translator.model.train_loss_track)), translator.model.train_loss_track,
             label="Train Loss")
    plt.title("Average Validation Loss over Training Iterations")
    plt.xlabel("Training Interations")
    plt.ylabel("Validation Loss")
    plt.legend()
    os.makedirs(f"./loss_overtime/run_id_{counter}", exist_ok=True)
    pd.DataFrame({"data_size": nrows, "alpha": learning_rate, "Epochs": epochs, "training_size": training_size,
                  "hidden_nodes": hidden_nodes,
                  "Avg. train_cross_entropy_loss": np.mean(
                      translator.model.train_loss_track),
                  "Avg. val_cross_entropy_loss": np.mean(
                      translator.model.val_loss_track),
                  "Translation": translator.execute_inference_pipeline(
                      statement='How are you doing')}, index=[0]).to_csv(f"./loss_overtime/logs.csv", mode='a', index=False) # for Translation change the statement parameter
    plt.savefig(f"./loss_overtime/run_id_{counter}/train_V_val_overtime.png")
    plt.clf()
    counter += 1
    # plt.show()


if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/ankitupadhyayutd/MLP/main/FRA.xlsx'
    fileName = wget.download(url)
    counter = 0
    combinations = []
    for nrows, sentance_length, training_size in [(15_000, 3, 64)]: # Parameters for various combinations [(25_000, 5, 128), (50_000, 6, 128)]
        for learning_rate in [0.005]: # Parameters for various combinations [0.0003, 0.0001]
            for epochs in [10, 25]:  # Parameters for various combinations [30, 70]
                for hidden_nodes in [100, 500]: # Parameters for various combinations [1000]
                    combinations.append(
                        (nrows, learning_rate, epochs, training_size, hidden_nodes))
    for nrows, learning_rate, epochs, training_size, hidden_nodes in combinations:
        train_and_infer(nrows, learning_rate, epochs, hidden_nodes,
                        training_size, counter, sentance_length, fileName)
        counter += 1
