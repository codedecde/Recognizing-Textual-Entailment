import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class recurrent_BatchNorm(nn.Module):
	def __init__(self, num_features, max_len, eps=1e-5, momentum=0.1, affine=True):
		super(recurrent_BatchNorm, self).__init__()
		self.num_features = num_features
		self.affine = affine
		self.max_len = max_len
		self.eps = eps
		self.momentum = momentum
		if self.affine:
			self.weight = nn.Parameter(torch.Tensor(num_features))
			self.register_parameter('weight', self.weight)			
			self.bias = nn.Parameter(torch.Tensor(num_features))
			self.register_parameter('bias', self.bias)
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		for i in xrange(max_len):
			self.register_buffer('running_mean_{}'.format(i), torch.zeros(num_features))
			self.register_buffer('running_var_{}'.format(i), torch.ones(num_features))		
		self.reset_parameters()

	def reset_parameters(self):
		for i in xrange(self.max_len):
			running_mean = getattr(self, 'running_mean_{}'.format(i))
			running_mean.zero_()
			running_var = getattr(self, 'running_var_{}'.format(i))
			running_var.fill_(1)        
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def _check_input_dim(self, input_, index):
		running_mean = getattr(self, 'running_mean_{}'.format(index))
		if input_.size(1) != running_mean.nelement():
			raise ValueError('got {}-feature tensor, expected {}'
			                 .format(input_.size(1), self.num_features))

	def forward(self, input_, index):
		if index >= self.max_len:
			index = self.max_len - 1
		self._check_input_dim(input_, index)
		running_mean = getattr(self, 'running_mean_{}'.format(index))
		running_var = getattr(self, 'running_var_{}'.format(index))
		return F.batch_norm(
			input_, running_mean, running_var, self.weight, self.bias,
			self.training, self.momentum, self.eps)

	def __repr__(self):
		return ('{name}({num_features}, eps={eps}, momentum={momentum},'
				' max_length={max_length}, affine={affine})'
				.format(name=self.__class__.__name__, **self.__dict__))