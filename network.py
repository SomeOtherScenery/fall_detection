import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    # num_inputs = 24
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        num_levels = len(num_channels)                      # num_channels = [8, 32, 64]
        tcn_blocks = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # Input size for each joint is 2 (x, y)
            out_channels = num_channels[i]
            tcn_blocks += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        # layers.append(nn.ModuleList(tcn_block))
        self.network = nn.ModuleList(tcn_blocks)

    def forward(self, x):
        for tcn_block in self.network:
            output = tcn_block(x)
            x = output
        return output                                                   # (1, 64, 18)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=18):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.register_parameter('positional_encoding', self.encoding)

    def forward(self, x):
        # batch_size, _, seq_length = x.size()
        seq_length, batch_size, _ = x.size()
        return x + self.encoding[:seq_length].unsqueeze(1).expand(-1, batch_size, -1)


class TCNTransformer(nn.Module):
    def __init__(self, num_inputs=24, d_model=32, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(TCNTransformer, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=num_inputs, num_channels=[8, 16, 32], kernel_size=3)
        self.d_model = d_model
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.fc_out = nn.Linear(32, 2)
        self.sf = nn.Softmax(dim=1)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.tcn(x)
        joint_feature = self.positional_encoding(x.permute(2, 0, 1))
        joint_feature = self.encoder(joint_feature)  # (seq, batch, feature) ---> (18, 1, 64)
        context = torch.mean(joint_feature, dim=0)
        out = self.sf(self.fc_out(context))
        return out, context


class TCNClassifier(nn.Module):
    def __init__(self, num_inputs=24):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=num_inputs, num_channels=[8, 16, 32], kernel_size=3)
        self.fc_out = nn.Linear(32, 2)
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.tcn(x)
        context = torch.mean(x, dim=2)
        out = self.sf(self.fc_out(context))
        return out, context


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=24, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.fc_out = nn.Linear(24, 2)  # Binary classification, one-hot encoding
        self.sf = nn.Softmax(dim=1)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        joint_feature = self.positional_encoding(x.permute(2, 0, 1))
        joint_feature = self.encoder(joint_feature)  # (seq, batch, feature) ---> (18, 1, 64)
        context = torch.mean(joint_feature, dim=0)
        out = self.sf(self.fc_out(context))
        return out, context


if __name__ == '__main__':
    model = TransformerClassifier()
    with torch.no_grad():
        model.eval()  # input_channel, num_input, kernel_size,
        test_data = torch.randn(1, 24, 18)
        output, _,  = model(test_data)
        print(output.shape)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable parameters: ", trainable_parameters)
        parameters = sum(p.numel() for p in model.parameters())
        print("all parameters: ", parameters)