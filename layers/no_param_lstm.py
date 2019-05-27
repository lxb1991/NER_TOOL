from torch.nn.modules import LSTM
from layers.param_gen import ParamGenerator
import torch


class NoParamLSTM(LSTM):

    def __init__(self, *args, **kwargs):
        super(NoParamLSTM, self).__init__(*args, **kwargs)
        self.param_gen = ParamGenerator(model_type=self.mode, num_layers=self.num_layers,
                                        input_size=self.hidden_size*2, hidden_size=self.hidden_size)

    def init_weight(self, task_vector, domain_vector):

        all_weight = self.param_gen(task_vector, domain_vector)
        assert self.mode == "LSTM"
        gate_size = 4 * self.hidden_size

        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                direction_key = "backward" if direction == 1 else "forward"
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * num_directions

                start_idx = 0
                w_ih = all_weight[layer][direction_key][start_idx: start_idx+gate_size * layer_input_size].\
                    view(gate_size, layer_input_size)

                start_idx += gate_size * layer_input_size
                w_hh = all_weight[layer][direction_key][start_idx: start_idx+gate_size * self.hidden_size].\
                    view(gate_size, self.hidden_size)

                start_idx += gate_size * self.hidden_size
                b_ih = all_weight[layer][direction_key][start_idx: start_idx+gate_size].view(gate_size)

                start_idx += gate_size
                b_hh = all_weight[layer][direction_key][start_idx: start_idx+gate_size].view(gate_size)

                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in weights]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

    def forward(self, input, hx=None, task_vector=None, domain_vector=None):
        # ref: https://www.cnblogs.com/hellcat/p/8509351.html to clear the exist parameters
        self.__dict__['_parameters'].clear()
        self.init_weight(task_vector, domain_vector)
        self.flatten_parameters()
        return super().forward(input, hx)
