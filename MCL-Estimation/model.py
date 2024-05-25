import torch.nn as nn


class MCLNet(nn.Module):

    def __init__(self, args):
        super(MCLNet, self).__init__()
        self.num_dofs = args.num_dofs
        self.num_output_samples = args.num_output_samples
        self.num_hidden_units = args.num_hidden_units
        self.torque_modulue = self._create_torque_layers()
        self.mcl_module = self._create_mcl_layers()
        self.linear = nn.Linear(self.num_hidden_units, 1)

    def _create_torque_layers(self):
        layers = []
        layers.append(nn.Conv1d(self.num_dofs, self.num_hidden_units, 3, stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm1d(self.num_hidden_units))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv1d(self.num_hidden_units, self.num_hidden_units, 3, stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm1d(self.num_hidden_units))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv1d(self.num_hidden_units, self.num_dofs, 3, stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm1d(self.num_dofs))
        return nn.Sequential(*layers)

    def _create_mcl_layers(self):
        layers = []
        layers.append(nn.Conv1d(self.num_dofs, self.num_hidden_units, 3, stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm1d(self.num_hidden_units))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv1d(self.num_hidden_units, self.num_hidden_units, 3, stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm1d(self.num_hidden_units))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool1d(2, stride=2))
        layers.append(nn.Conv1d(self.num_hidden_units, self.num_hidden_units, 3, stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm1d(self.num_hidden_units))
        layers.append(nn.ReLU(inplace=False))
        return nn.Sequential(*layers)

    def forward(self, angle, acc):
        torque_passive = self.torque_modulue(angle)
        torque_active = acc - torque_passive
        mcl = self.mcl_module(torque_active)
        mcl = mcl.transpose(1, 2)
        mcl = self.linear(mcl)
        mcl = mcl.transpose(1, 2)
        mcl = mcl.view(mcl.size(0), self.num_output_samples)
        return mcl


if __name__ == "__main__":
    pass
