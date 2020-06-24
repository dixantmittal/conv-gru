import torch as t


class ConvGRUCell(t.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias):
        super().__init__()
        self.z = t.nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias)
        self.r = t.nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias)
        self.n = t.nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias)

    def forward(self, input, history):
        phi = t.cat((input, history), dim=1)
        z = t.sigmoid(self.z(phi))
        r = t.sigmoid(self.r(phi))

        phi = t.cat((input, r * history), dim=1)
        n = t.tanh(self.n(phi))

        return z * n + (1 - z) * history


class ConvGRU(t.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_layers=1, bias=True, batch_first=False):
        super().__init__()

        self.layers = t.nn.ModuleList()
        self.layers.append(ConvGRUCell(input_channels, output_channels, kernel_size, bias))
        for i in range(num_layers - 1):
            self.layers.append(ConvGRUCell(output_channels, output_channels, kernel_size, bias))

        self.num_layers = num_layers
        self.hidden_channels = output_channels
        self.batch_first = batch_first

    # input is trajectory first
    def forward(self, input: t.FloatTensor, hidden=None):
        # permute input to time first if batch first is true
        if self.batch_first:
            input = input.permute(1, 0, 2, 3, 4)

        # init hidden if None
        if hidden is None:
            _, b, _, h, w = input.shape
            hidden = t.zeros(self.num_layers, b, self.hidden_channels, h, w).to(self.layers[0].z.weight.device)

        output_hidden = []
        layer_input = input
        for h, gru_layer in zip(hidden, self.layers):
            # pass through the sequence
            seq_output = []
            for i in layer_input:
                h = gru_layer(i, h)
                seq_output.append(h)
            # store the last hidden tensor
            output_hidden.append(h)
            layer_input = seq_output

        return t.stack(layer_input), t.stack(output_hidden)
