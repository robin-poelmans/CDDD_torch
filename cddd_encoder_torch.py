import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import argparse

class CustomGRUCell(nn.Module):
	"""
	GRU cell implementation following the original paper formulation where
	the reset gate is applied before the matrix multiplication.
	"""

	def __init__(self, input_size, hidden_size, bias = True):
		super(CustomGRUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
		self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))

		if bias:
			self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))

		else:
			self.register_parameter('bias_ih', None)

	def reset_parameters(self):
		"""
		Initialize parameters using the same method as PyTorch's GRU
		"""

		std = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-std, std)

	def set_weights_from_numpy(self, weight_ih, weight_hh, bias_ih = None):
		self.weight_ih.data = torch.from_numpy(weight_ih).float()
		self.weight_hh.data = torch.from_numpy(weight_hh).float()

		if bias_ih is not None:
			self.bias_ih.data = torch.from_numpy(bias_ih).float()

	def forward(self, input, hidden):
		"""
		Implements the original GRU formulation where reset gate is applied
		before matrix multiplication.

		Args:
			input: tensor of shape (batch, input_size)
			hidden: tensor of shape (batch, hidden_size)

		Returns:
			next_hidden: tensor of shape (batch, hidden_size)
		"""

		batch_size = input.size(0)
		if hidden is None:
			hidden = torch.zeros(batch_size, self.hidden_size, device = input.device, dtype = input.dtype)

		# Split weights for update gate (z), reset gate (r) and new gate (n)
		w_ih_chunks = self.weight_ih.chunk(3, 0)
		w_hh_chunks = self.weight_hh.chunk(3, 0)

		if self.bias_ih is not None:
			b_ih_chunks = self.bias_ih.chunk(3,0)
		else:
			b_ih_chunks = (None, None, None)

		# Update gate
		z_t = torch.sigmoid(
			torch.mm(input, w_ih_chunks[0].t()) + 
			torch.mm(hidden, w_hh_chunks[0].t()) +
			(b_ih_chunks[0] if b_ih_chunks[0] is not None else 0)
			)

		# Reset gate
		r_t = torch.sigmoid(
			torch.mm(input, w_ih_chunks[1].t()) +
			torch.mm(hidden, w_hh_chunks[1].t()) +
			(b_ih_chunks[1] if b_ih_chunks[1] is not None else 0)
			)

		# New gate (using original formulation)
		# First apply reset gate to hidden state = element-wise multiplication

		reset_hidden = r_t * hidden

		# Then apply weight matrix to reset hidden state
		n_t = torch.tanh(
			torch.mm(input, w_ih_chunks[2].t()) +
			torch.mm(reset_hidden, w_hh_chunks[2].t()) +
			(b_ih_chunks[2] if b_ih_chunks[2] is not None else 0)
			)

		# Compute next hidden state
		h_next = (1 - z_t) * n_t + z_t * hidden # Elementwise products

		return h_next

class CustomGRU(nn.Module):
	"""
	Multi-layer GRU with original paper's formulation.
	Arguments:
		hidden_sizes: list-like object of length = num_layers
	"""

	def __init__(self, input_size, hidden_sizes, num_layers = 1, bias = True, batch_first = True, dropout = 0):
		super(CustomGRU, self).__init__()
		self.input_size = input_size
		self.num_layers = num_layers
		if isinstance(hidden_sizes, int):
			hidden_sizes = [hidden_sizes]
		self.hidden_sizes = hidden_sizes
		self.bias = bias
		self.batch_first = batch_first
		self.dropout = dropout

		# Create a list of GRU cells
		self.cells = nn.ModuleList([
			CustomGRUCell(
				self.input_size if layer == 0 else self.hidden_sizes[layer - 1],
				self.hidden_sizes[layer],
				bias = bias
				)
			for layer in range(num_layers)
		])

		self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

	def set_weights_from_numpy(self, layer_weights):
		"""
		Set weights for all layers from numpy arrays.
		Args:
			layer_weights: list of dictionaries, one per layer, each containing:
				'weight_ih', 'weights_hh', 'bias_ih'
		"""

		for layer_idx, weights_dict in enumerate(layer_weights):
			self.cells[layer_idx].set_weights_from_numpy(
				weights_dict['weights_ih'], weights_dict['weights_hh'],
				bias_ih = weights_dict.get('bias_ih')
			)

	def forward(self, input, mask = None, hidden = None):
		"""
		Args:
			input: tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first=True
			hidden: tensor of shape (num_layers, batch, hidden_size)

		Returns:
			output: tensor of shape (seq_len, batch, hidden_size) or (batch, seq_len, hidden_size) if batch_first=True
			hidden: tensor of shape (num_layers, batch, hidden_size)
		"""

		if self.batch_first:
			input = input.transpose(0, 1)

		seq_len, batch_sz, _ = input.size()

		if hidden is None:
			hidden = [torch.zeros(batch_sz, hidden_size, device = input.device, dtype = input.dtype) for hidden_size in self.hidden_sizes]

		output = [[] for _ in range(self.num_layers)] # Instantiate 3 empty lists
		layer_states = hidden

		# Process each timestep
		for t in range(seq_len):
			inp = input[t]
			layer_output = inp

			# Process each layer
			for layer in range(self.num_layers):
				layer_hidden = layer_states[layer]

				if layer > 0 and self.dropout_layer:
					layer_output = self.dropout_layer(layer_output)

				layer_output = self.cells[layer](layer_output, layer_hidden)
				layer_states[layer] = layer_output
				output[layer].append(layer_output)

		mask_expanded = mask.unsqueeze(-1)

		# Simply taking final hidden state is incorrect, since we work with padded sequences
		hidden_layers_concat = []
		for layer_idx, layer_output in enumerate(output):
			masked_output = torch.stack(layer_output).transpose(0, 1) * mask_expanded
			result = masked_output.sum(dim = 1)
			hidden_layers_concat.append(result)
		hidden_concat = torch.cat(hidden_layers_concat, dim = 1) # Concatenate 3 hidden layers

		return hidden_concat

class CDDD_encoder(nn.Module):
	def __init__(self, batch_size, weights_dir):
		super(CDDD_encoder, self).__init__()

		self.weights_dir = weights_dir
		self.embedding_sizes = [32, 512, 1024, 2048]

		# Initial embedding layer
		char_weights = np.load(f'{self.weights_dir}/char_embedding_0.npy')
		self.char_projection = nn.Linear(40, self.embedding_sizes[0], bias=False)
		self.char_projection.weight.data = torch.FloatTensor(char_weights.T)

		# 3 GRU layers
		self.multi_layer_gru = CustomGRU(self.embedding_sizes[0], self.embedding_sizes[1:], num_layers = 3)

		# Final dense layer
		dense_kernel = np.load(f'{self.weights_dir}/Encoder_dense_kernel_0.npy')
		self.dense = nn.Linear(sum(self.embedding_sizes[1:]), dense_kernel.shape[1])
		self.tanh = nn.Tanh()

		self._load_gru_weights()
		self._load_dense_weights()

	def _load_gru_weights(self):

		list_of_dicts = []

		for i in range(3):
			gates_kernel = np.load(f'{self.weights_dir}/Encoder_rnn_multi_rnn_cell_cell_{i}_gru_cell_gates_kernel_0.npy')
			gates_bias = np.load(f'{self.weights_dir}/Encoder_rnn_multi_rnn_cell_cell_{i}_gru_cell_gates_bias_0.npy')
			candidate_kernel = np.load(f'{self.weights_dir}/Encoder_rnn_multi_rnn_cell_cell_{i}_gru_cell_candidate_kernel_0.npy')
			candidate_bias = np.load(f'{self.weights_dir}/Encoder_rnn_multi_rnn_cell_cell_{i}_gru_cell_candidate_bias_0.npy')

			input_size = self.embedding_sizes[i]
			hidden_size = self.embedding_sizes[i+1]

			gates_kernel_i = gates_kernel[:input_size, :]
			gates_kernel_h = gates_kernel[input_size:, :]
			candidate_kernel_i = candidate_kernel[:input_size, :]
			candidate_kernel_h = candidate_kernel[input_size:, :]

			w_ih = np.concatenate([
				gates_kernel_i[:, hidden_size:],
				gates_kernel_i[:, :hidden_size],
				candidate_kernel_i], axis = 1).T

			w_hh = np.concatenate([
				gates_kernel_h[:, hidden_size:],
				gates_kernel_h[:, :hidden_size],
				candidate_kernel_h], axis = 1).T

			b_ih = np.concatenate([
				gates_bias[hidden_size:],            # reset gate
				gates_bias[:hidden_size],            # update gate
				candidate_bias])

			weights_dict = {}
			weights_dict['weights_ih'] = w_ih
			weights_dict['weights_hh'] = w_hh
			weights_dict['bias_ih'] = b_ih

			list_of_dicts.append(weights_dict)

		self.multi_layer_gru.set_weights_from_numpy(list_of_dicts)

	def _load_dense_weights(self):
		dense_kernel = np.load(f'{self.weights_dir}/Encoder_dense_kernel_0.npy')
		dense_bias = np.load(f'{self.weights_dir}/Encoder_dense_bias_0.npy')
		self.dense.weight.data = torch.FloatTensor(dense_kernel.T)
		self.dense.bias.data = torch.FloatTensor(dense_bias)

	def forward(self, x, mask = None):
		x = self.char_projection(x)
		x = self.multi_layer_gru(x, mask = mask)
		x = self.dense(x)
		x = self.tanh(x)
		return x

class CDDD_runner:
	"""
	Class for the tokenization of SMILES sequences
	"""
	def __init__(self, args):
		"""
		Input:
			args: input_path, output_path, weights_dir, batch_size
		"""

		og_dict = {0: '</s>', 1: '#', 2: '%', 3: ')', 4: '(', 5: '+', 6: '-', 7: '1', 
		8: '0', 9: '3', 10: '2', 11: '5', 12: '4', 13: '7', 14: '6', 15: '9', 
		16: '8', 17: ':', 18: '=', 19: '@', 20: 'C', 21: 'B', 22: 'F', 23: 'I', 
		24: 'H', 25: 'O', 26: 'N', 27: 'P', 28: 'S', 29: '[', 30: ']', 31: 'c', 
		32: 'i', 33: 'o', 34: 'n', 35: 'p', 36: 's', 37: 'Cl', 38: 'Br', 39: '<s>'}

		self.vocab_dict = {value: key for key, value in og_dict.items()}

		self.batch_size = args.batch_size
		self.output_path = args.o
		self.weights_dir = args.weights_dir
		with open(args.i, 'r') as infile:
			smiles_list = [line.strip().replace('/', '').replace('\\', '') for line in infile] # Remove all stereochemistry information

		self.smiles_list = sorted(smiles_list, key = len) # Sort based on length to optimize padding performance

	def tokenize_batch(self, smiles_batch):
		"""
		Input:
			smiles_batch: list of SMILES strings, length equal to batch_size
		Output:
			input_tensor: torch.tensor(batch_size, sequence_length, num_features). One-hot encoding of the features
			mask: torch.tensor(batch_size, sequence_length). Shows length of original sequences before padding
		"""

		batch_size = len(smiles_batch)
		max_length = len(smiles_batch[-1]) + 2 # Last entry is the largest after sorting. +2 to account for start and stop token
		num_features = len(self.vocab_dict)
		input_tensor = torch.zeros(batch_size, max_length, num_features)
		mask = torch.zeros(batch_size, max_length)

		for batch_idx, smiles in enumerate(smiles_batch):
			tokens = []
			i = 0
			while i < len(smiles):
				if i < len(smiles) - 1:
					two_chars = smiles[i:i+2]
					if two_chars in ['Cl', 'Br']:
						tokens.append(self.vocab_dict[two_chars])
						i += 2
						continue
				if smiles[i] in self.vocab_dict:
					tokens.append(self.vocab_dict[smiles[i]])
				i += 1
			tokens = [self.vocab_dict['<s>']] + tokens + [self.vocab_dict['</s>']]
			for i, token_idx in enumerate(tokens):
				input_tensor[batch_idx, i, token_idx] = 1

			mask[batch_idx, len(tokens) - 1] = 1

		return input_tensor, mask

	def main_process(self):
		model = CDDD_encoder(self.batch_size, self.weights_dir)
		model.eval()
		model = model.to('cpu')

		smiles_batch_list = [self.smiles_list[i:i+self.batch_size] for i in range(0, len(self.smiles_list), self.batch_size)]

		embeddings_list = []

		with torch.no_grad():
			for smiles_batch in smiles_batch_list:
				input_tensor, mask = self.tokenize_batch(smiles_batch)
				embeddings = model(input_tensor, mask = mask).numpy() # Convert tensor to numpy array
				embeddings_list.append(embeddings)

		# Concatenate all numpy arrays
		embeddings_array = np.concatenate(embeddings_list, axis = 0)
		feature_columns = [f'cddd_{i+1}' for i in range(embeddings_array.shape[1])]
		df = pd.DataFrame(
			data = embeddings_array,
			index = self.smiles_list,
			columns = feature_columns)

		df.to_csv(self.output_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type = str, help = 'Input file in .smi format. One string per row, row should only contain SMILES')
	parser.add_argument('-o', type = str, help = 'Path to output csv file')
	parser.add_argument('--weights_dir', type = str, help = 'Path to directory containing weights for all layers in .npy format')
	parser.add_argument('--batch_size', type = int, help = 'Batch size for the encoder')
	args = parser.parse_args()
	cddd_runner = CDDD_runner(args)
	cddd_runner.main_process()
