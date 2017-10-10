import random
import math

import numpy as np
import matplotlib.pyplot as plt


#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

class NeuralNetwork:
    LEARNING_RATE = 0.

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer1 = NeuronLayer(3, hidden_layer_bias)
        self.hidden_layer2 = NeuronLayer(2, hidden_layer_bias)

        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)

        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer1.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer1.neurons[h].weights.append(random.uniform(-0.5, 0.5))
                else:
                    self.hidden_layer1.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

        weight_num = 0
        for h in range(len(self.hidden_layer2.neurons)):
            for i in range(len(self.hidden_layer1.neurons)):
                if not hidden_layer_weights:
                    self.hidden_layer2.neurons[h].weights.append(random.uniform(-0.5, 0.5))
                else:
                    self.hidden_layer2.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer2.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.uniform(-0.5, 0.5))
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1



    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer1_outputs = self.hidden_layer1.feed_forward(inputs)
        hidden_layer2_outputs = self.hidden_layer2.feed_forward(hidden_layer1_outputs)

        # --------------------- can make a chain here ------------------------
        return self.output_layer.feed_forward(hidden_layer2_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas initialization
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # dE/ dWjk : deltaK
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # -----------------------   could add looping for n - hidden layers   -----------------------------------
        # 2. Hidden neuron deltas : deltaJ
        pd_errors_wrt_hidden2_neuron_total_net_input = [0] * len(self.hidden_layer2.neurons)
        for h in range(len(self.hidden_layer2.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron

            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                #deltaL* Wij
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h] #sum(deltaK * Wjk)

            # deltaJ = Oj * (1 - Oj) * sum(deltaK * Wjk)
            pd_errors_wrt_hidden2_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer2.neurons[h].calculate_pd_total_net_input_wrt_input()


        pd_errors_wrt_hidden1_neuron_total_net_input = [0] * len(self.hidden_layer1.neurons)
        for h in range(len(self.hidden_layer1.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron

            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.hidden_layer2.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_hidden2_neuron_total_net_input[o] * self.hidden_layer2.neurons[o].weights[h]  # sum(deltaK * Wjk)

            # deltaJ = Oj * (1 - Oj) * sum(deltaK * Wjk)
            pd_errors_wrt_hidden1_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer1.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # P_ = deltaK times Oj
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                # Wjk = Wjk - n * P_
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer2.neurons)):
            for w_ih in range(len(self.hidden_layer2.neurons[h].weights)):
                # P_ = deltaJ times Oi
                pd_error_wrt_weight = pd_errors_wrt_hidden2_neuron_total_net_input[h] * self.hidden_layer2.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                # Wij = Wij - n * P_
                self.hidden_layer2.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer1.neurons)):
            for w_ih in range(len(self.hidden_layer1.neurons[h].weights)):
                # P_ = deltaJ times Oi
                pd_error_wrt_weight = pd_errors_wrt_hidden1_neuron_total_net_input[h] * self.hidden_layer1.neurons[
                    h].calculate_pd_total_net_input_wrt_weight(w_ih)
                # Wij = Wij - n * P_
                self.hidden_layer1.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
                # plt.plot(total_error)
                # plt.ylabel("Sigmoid function")
                # plt.xlabel('Range of x')
                # plt.show()

        return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else 1

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #

    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

###

# Blog post example:

# nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_weightsdden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
# for i in range(10000):
#     nn.train([0.05, 0.1], [0.01, 0.99])
#     print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# XOR example:

training_sets = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

num_of_inputs = len(training_sets[0][0])
num_of_outputs = len(training_sets[0][1])
num_of_hidden_units = 4


hidden_layer_bias = 1.27546647732
output_layer_bias = 0.670661698408
hidden_layer_weights = [0.24224589619972614, 0.47589624529986785, -0.40746440711877696, 0.4855313370435048, -0.35524491152598103, -0.26786164415951175, 0.0005698297643920158, -0.20072083117186879]
output_layer_weights = [-0.3460655110385422, -0.40625244341523403, 0.4113354296243792, -0.4245191770024349, 0.47232777559508743, 0.4347981906743703, 0.36284324430203707, -0.2511815553956299]
nn = NeuralNetwork(num_of_inputs, num_of_hidden_units, num_of_outputs, hidden_layer_weights, hidden_layer_bias,output_layer_weights,output_layer_bias)

err = []
for i in range(300000):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)
    total_err = nn.calculate_total_error(training_sets)
    print(i, total_err)
    err.append(total_err)


plt.plot(err)
plt.ylabel("Error")
plt.xlabel('Iterations')
plt.savefig('Q5a.png')
plt.close()