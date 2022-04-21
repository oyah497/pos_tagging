from pathlib import Path
from typing import Dict, List

import sys, os
# sys.path.append(os.pardir)
#sys.path.append(os.path.abspath('..'))

import numpy as np
from collections import OrderedDict
from ..common.layers import *
from ..common.gradient import numerical_gradient

from .model import Model

PARAMETER_FILE_NAME = "parameters.npz"


@Model.register("neural_network")
class NeuralNetwork(Model):
    def __init__(self, num_features: int, num_classes: int, hidden_size_list = [100, 100],  wordvec_size: int = 100, activation='relu', weight_init_std=0.01, learning_rate: float = 0.001):
        super().__init__(num_features, num_classes)
        self.wordvec_size = wordvec_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.learning_rate = learning_rate
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()

        self.layers['Embedding'] = Embedding(self.params['Emb'])

        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
        
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()


    def __init_weight(self, weight_init_std):
        all_size_list = [self._num_features] + [self.wordvec_size] + self.hidden_size_list + [self.num_classes]
        self.params['Emb'] = 0.01 * np.random.randn(all_size_list[0], all_size_list[1])

        for idx in range(2, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
            
            self.params['W' + str(idx-1)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx-1)] = np.zeros(all_size_list[idx])

    def output(self, word_features: List[List[int]]):
        x = word_features
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x


    def predict(self, word_features: List[List[int]]) -> List[int]:
        output = self.output(word_features)
        return output.argmax(axis=1).tolist()

    
    def loss(self, x, t):
        y = self.output(x)
        return self.last_layer.forward(y, t), y


    def gradient(self, x, t):
        loss, output = self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['Emb'] = self.layers['Embedding'].dW
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        
        return grads, loss, output



    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        num_words = len(tags)
        #print(num_words)
        tags_one_hot = np.zeros(shape=(num_words, self.num_classes))
        for i in range(num_words):
            tags_one_hot[i][tags[i]] = 1
        
        grads, loss, output = self.gradient(word_features, tags_one_hot)
        predicted_labels = output.argmax(axis=1).tolist()

        for key in grads.keys():
            self.params[key] -= self.learning_rate * grads[key]
        
        return {"prediction": predicted_labels}

        """
        predicted_labels = self.predict(word_features)

        incorrect_indices = [
            i for i, (ground_truth, prediction) in enumerate(zip(tags, predicted_labels)) if ground_truth != prediction
        ]

        for i in incorrect_indices:
            ground_truth_tag_idx = tags[i]
            prediction_tag_idx = predicted_labels[i]
            features = word_features[i]

            self.weights[:, ground_truth_tag_idx][features] += 1
            self.bias[ground_truth_tag_idx] += 1
            self.weights[:, prediction_tag_idx][features] -= 1
            self.bias[prediction_tag_idx] -= 1

        return {"prediction": predicted_labels}
        """

    def save(self, save_directory: str):
        np.savez(Path(save_directory) / PARAMETER_FILE_NAME, **(self.params))

    @classmethod
    def load(cls, save_directory: str) -> "NeuralNetwork":
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, wordvec_size = parameters['Emb'].shape
        hidden_size_list = []
        i = 2
        while(True):
            if ('b' + str(i)) in parameters:
                hidden_size_list.append(parameters['b' + str(i-1)].shape[0])
            else:
                break
            i += 1
        num_classes = parameters['b' + str(i-1)].shape[0]
        model = NeuralNetwork(num_features, num_classes, hidden_size_list, wordvec_size)
        for key in model.params.keys():
            model.params[key] = parameters[key]
        return model
        """
        num_features, num_classes = parameters["weights"].shape
        model = NeuralNetwork(num_features, num_classes)
        model.weights = parameters["weights"]
        model.bias = parameters["bias"]
        return model

        """
