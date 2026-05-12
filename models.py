import torch
import torch.nn as nn

class TinyCNN(nn.Module):
	def __init__(self, nLabels):
		super(TinyCNN, self).__init__()

		self.model = nn.Sequential(
			nn.Conv1d(1, 10, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(10),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(10, 20, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(20),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(20, 40, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(40),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(40, 10, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(10),
			nn.Dropout(0.1),
			nn.AvgPool1d(2),

			nn.Conv1d(10, 12, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm1d(12),
			nn.Dropout(0.2),

			nn.Conv1d(12, 10, kernel_size=1),
			nn.Dropout(0.2),

			nn.Linear(300, 32), # batch, filters, * -> batch, filters, 32
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(10*32, 128),
			nn.ReLU(),
			nn.Linear(128, nLabels)
		)

	def forward(self, x):
		# x = torch.unsqueeze(x,1)
		x = self.model(x)
		return x

class Tiny_NLL_CNN(nn.Module):
	def __init__(self, nLabels):
		super(Tiny_NLL_CNN, self).__init__()

		self.model = nn.Sequential(
			nn.Conv1d(1, 10, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(10),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(10, 20, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(20),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(20, 40, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(40),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(40, 10, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(10),
			nn.Dropout(0.1),
			nn.AvgPool1d(2),

			nn.Conv1d(10, 12, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm1d(12),
			nn.Dropout(0.2),

			nn.Conv1d(12, 10, kernel_size=1),
			nn.Dropout(0.2),

			nn.Linear(300, 32), # batch, filters, * -> batch, filters, 32
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(10*32, 128),
			nn.ReLU(),
			nn.Linear(128, 2*nLabels)
		)

	def forward(self, x):
		# x = torch.unsqueeze(x,1)
		x = self.model(x)
		return x

class Medium_NF_CNN(nn.Module):
	def __init__(self, num_params):
		super(Medium_NF_CNN, self).__init__()

		self.model = nn.Sequential(
			nn.Conv1d(1, 10, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(10),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(10, 20, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(20),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(20, 40, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(40),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(40, 80, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm1d(80),
			nn.Dropout(0.1),
			nn.AvgPool1d(3),

			nn.Conv1d(80, 20, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(20),
			nn.Dropout(0.1),
			nn.AvgPool1d(2),

			nn.Conv1d(20, 24, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm1d(24),
			nn.Dropout(0.2),

			nn.Conv1d(24, 20, kernel_size=1),
			nn.Dropout(0.2),

			nn.AdaptiveAvgPool1d(4),
			nn.Flatten(),

			nn.Linear(80, 128),
			nn.ReLU(),
			nn.Linear(128, num_params)
		)

	def forward(self, x):
		# x = torch.unsqueeze(x,1)
		x = self.model(x)
		return x

class TinyCNNEncoder(nn.Module):
    def __init__(self, input_length, num_params):
        super(TinyCNNEncoder, self).__init__()

        self.model = nn.Sequential(
			nn.Flatten(),
            nn.Linear(input_length, num_params),
        )

    def forward(self, x):
        x = self.model(x)
        return x