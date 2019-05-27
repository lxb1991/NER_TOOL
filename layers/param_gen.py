import torch
import math
from torch.nn import Module
from torch.nn import init
from torch.nn import Parameter


class ParamGenerator(Module):

    def __init__(self, model_type, num_layers, input_size, hidden_size):
        super(ParamGenerator, self).__init__()
        self.task_vector_dimension = 8
        self.domain_vector_dimension = 8
        self.model_type = model_type
        self.num_layers = num_layers
        if self.model_type == 'LSTM':
            self.meta_matrix = self.init_meta_by_model_type(input_size, hidden_size)
        elif self.model_type == 'LINEAR':
            self.meta_matrix = self.init_meta_by_model_type(input_size, hidden_size)
        self.reset_param(hidden_size)
        print(self.model_type + ' param generator init down')

    def init_meta_by_model_type(self, input_size, hidden_size):
        cube_param = {}
        if self.model_type == 'LSTM':
            # 4 * ( hidden * input + hidden + hidden * hidden + hidden)
            flatten_dimension = 4 * (hidden_size * input_size + hidden_size * hidden_size + hidden_size * 2)
            for layer in range(self.num_layers):
                cube_param[layer] = {}
                for direction in range(2):
                    direction_key = "backward" if direction == 1 else "forward"
                    cube_param[layer][direction_key] = Parameter(torch.Tensor(self.task_vector_dimension,
                                                                              flatten_dimension,
                                                                              self.domain_vector_dimension))
                    setattr(self, "meta_param_" + str(layer) + "_" + direction_key, cube_param[layer][direction_key])
        elif self.model_type == "LINEAR":
            for layer in range(self.num_layers):
                # input * output + output
                flatten_dimension = input_size * hidden_size + hidden_size
                cube_param[layer] = Parameter(torch.Tensor(self.task_vector_dimension, flatten_dimension,
                                                           self.domain_vector_dimension))
                setattr(self, "meta_param_" + str(layer) + "_linear", cube_param[layer])

        return cube_param

    def reset_param(self, hidden_size):
        stdv = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, task_vector, domain_vector):
        weight = {}
        if self.model_type == 'LSTM':
            for layer in range(self.num_layers):
                weight[layer] = {}
                for direction in range(2):
                    direction_key = "backward" if direction == 1 else "forward"

                    task_emb = torch.matmul(task_vector,
                                            self.meta_matrix[layer][direction_key].view(self.task_vector_dimension, -1))
                    weight[layer][direction_key] = torch.matmul(task_emb.view(-1, self.domain_vector_dimension),
                                                                domain_vector.view(self.domain_vector_dimension, 1))
        elif self.model_type == "LINEAR":
            for layer in range(self.num_layers):
                weight[layer] = torch.matmul(torch.matmul(task_vector,
                                                          self.meta_matrix[layer].view(self.task_vector_dimension, -1)).
                                             view(-1, self.domain_vector_dimension),
                                             domain_vector.view(self.domain_vector_dimension, 1))

        return weight
