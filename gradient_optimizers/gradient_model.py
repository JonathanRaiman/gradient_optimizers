from utils.nnet_named import NamedGradientStatus
from utils.nnet import symbolic_variable_for_dimension
from numpy import array
import theano, theano.tensor as T, numpy as np
from collections import OrderedDict

REAL = theano.config.floatX

class GradientModel(object):
	"""
	GradientModel
	-------------
	
	A gradient model for updating your model with
	hessian free, adagrad, or linear decay updates.

	You will need to define the following attributes,
	and fill them as appropriate:

		self.params = []
    	self.indexed_params = set()

    	self._l2_regularization = True / False

    	# if L2 is true store this parameter:
        self._l2_regularization_parameter = theano.shared(np.float64(l2_regularization).astype(REAL), name='l2_regularization_parameter')

    Upon initialization you must run:

    	self._select_update_mechanism(update_method_name)

    	# then to compile this mechanism:
    	self.create_update_fun()


    The update methods expect the input to be of the form:

    	ivector <indices>, ivector <labels>

    If this is not the case you can modify them as appropriate.

	"""

	def alternate_gradients_fun(self, params, example):
		raise NotImplementedError()

	def loss_fun(self, examples):
		raise NotImplementedError()

	def gradients_fun(self, example):
		raise NotImplementedError()

	def create_update_mechanisms(self):
		self.update_mechanisms = {}
		self._param_shapes     = {}

		grad_update   = theano.tensor.vector('grad_update')
		learning_rate = theano.tensor.vector('learning_rate')

		mixed_rate = -grad_update * learning_rate

		for param in self.params:
			shape                         = param.get_value(borrow = True).shape
			self._param_shapes[param]     = shape
			self.update_mechanisms[param] = theano.function([grad_update, learning_rate],
				updates = (
					{param: param + mixed_rate.reshape(shape)} if len(shape) > 0 else {param: param + mixed_rate[0]}) )

	def get_parameter(self, param):
		"""
		Get the value of a parameter.

		Inputs:
		-------

				 param: the parameter.

		Outputs:
		--------

		 ndarray with the underlying parameter value.

		"""
		return param.get_value(borrow = True)

	def update_parameter(self, param, grad, learning_rate):
		"""
		Make updates to parametes, either the tensors, matrices, or the
		vocabulary.

		Inputs:
		-------

				 param: the parameter.
				  grad: the ndarray of the gradient for the error of this parameter
		 learning_rate: the per dimension learning rate for this grad (i.e.
						len(grad) == len(learning_rate))

		"""
		self.update_mechanisms[param](grad, learning_rate)

	def unravel_parameter(self, param, vector):
		"""
		Reconvert parameter from its vectorized shape to its
		structured tensor or matrix shape.

		Inputs:
		-------

		  param: the underlying parameter
		 vector: the vectorized form of the parameter

		Outputs:
		-------

		 ndarray with shape equal to the parameter's and the
		 contents of vector.

		"""
		return vector.reshape(self._param_shapes[param])

	def gradients(model):
		gradients = NamedGradientStatus()
		gradients.add_symbolic_variables(model.params)
		return gradients

	def ravel_gradient(self, param, gradients):
		"""
		Convert parameter from its original shape to a
		vectorized form.

		Inputs:
		-------

			 param: the underlying parameter (Theano SharedVariable)
		 gradients: the NamedGradientStatus object containing
					gradient updates.

		Outputs:
		-------

		 ndarray vector ndim == 1 and the
		 contents of appropriate gradient update.

		"""
		return gradients[param].ravel()

	def ravel_gradients(self, param, gradients):
		return array([self.ravel_gradient(param, grad) for grad in gradients])

	def _skip_update_param(self, parameter):
		"""
		Whether to update this parameter given the current
		parametrization of the model.

		"""
		return False

	def _create_linear_batch_update_mechanism(self):
		indices              = T.ivector('indices')
		index_length         = T.ivector('indices_length')
		label                = T.ivector('labels')

		multi_class_projections = self.batch_projection_function(indices, index_length)

		batch_cost = self.batch_cost_function(multi_class_projections, label).mean()

		batch_gparams = T.grad(batch_cost, self.params, disconnected_inputs = self.disconnected_inputs )
		batch_updates = OrderedDict()
		
		for param, gparam in zip(self.params, batch_gparams):

			if self._skip_update_param(param):
				continue

			if param in self.indexed_params:
				batch_updates[param] = T.inc_subtensor(param[indices], - self.learning_rate * gparam[indices])
			else:
				batch_updates[param] = param - self.learning_rate * gparam

		self.batch_update_fun    = theano.function([indices, index_length, label], batch_cost, updates = batch_updates, mode = self.theano_mode)
		self.batch_gradient_fun  = theano.function([indices, index_length, label], batch_gparams, mode = self.theano_mode)
		self.predict_batch_proba = theano.function([indices, index_length], multi_class_projections, mode = self.theano_mode)
		self.predict_batch       = theano.function([indices, index_length], multi_class_projections.argmax(axis=0), mode = self.theano_mode)

	def l2_regularization(self, indices):
		costs = []
		for param in self.params:
			if param in self.indexed_params:
				costs.append(self._l2_regularization_parameter * (param[indices] ** 2).sum())
			else:
				costs.append(self._l2_regularization_parameter * (param ** 2).sum())
		return sum(costs)

	def _create_linear_update_mechanism(self):
		indices              = T.ivector('indices')
		label                = T.ivector('labels')
		
		class_projection     = self.projection_function(indices)
		
		cost                 = self.cost_function(class_projection, label).sum()

		if self._l2_regularization:
			cost += self.l2_regularization(indices)
		
		gparams              = T.grad(cost, self.params, disconnected_inputs  = self.disconnected_inputs)
		updates              = OrderedDict()
		
		for param, gparam in zip(self.params, gparams):

			if self._skip_update_param(param):
				continue

			if param in self.indexed_params:
				updates[param] = T.inc_subtensor(param[indices], - self.learning_rate * gparam[indices])
			else:
				updates[param] = param - self.learning_rate * gparam

		self.update_fun      = theano.function([indices, label], cost, updates = updates, mode = self.theano_mode)
		self.gradient_fun    = theano.function([indices, label], gparams, mode = self.theano_mode)
		# to compute hessian approximations, we use alternate gradient:
		alt_params           = [symbolic_variable_for_dimension(param.ndim) for param in self.params]
		givens               = OrderedDict()
		for param, alt_param in zip(self.params, alt_params):
			givens[param]    = alt_param

		self.alternate_gradient_fun = theano.function([indices, label] + alt_params, gparams, givens = givens, mode = self.theano_mode)

		if self.create_batch_methods:
			self._create_linear_batch_update_mechanism()

	def _create_clipped_update_mechanism(self):

		self.max_update_size = theano.shared(np.zeros(len(self.params), dtype=REAL), 'max_update_size')

		self.clip_range      = theano.shared(np.float32(10))
		indices              = T.ivector('indices')
		label                = T.ivector('labels')
		
		class_projection     = self.projection_function(indices)
		
		cost                 = self.cost_function(class_projection, label).sum()
		
		gparams              = T.grad(cost, self.params, disconnected_inputs = self.disconnected_inputs )
		updates              = OrderedDict()
		
		i = 0
		updates[self.max_update_size] = self.max_update_size
		for param, gparam in zip(self.params, gparams):

			if self._skip_update_param(param):
				continue

			if param in self.indexed_params:
				updates[self.max_update_size] = T.set_subtensor(updates[self.max_update_size][i], T.maximum(self.max_update_size[i], gparam[indices].max()))
				gparam = T.clip(gparam[indices], -self.clip_range, self.clip_range)
				updates[param] = T.inc_subtensor(param[indices], - self.learning_rate * gparam)
			else:
				updates[self.max_update_size] = T.set_subtensor(updates[self.max_update_size][i], T.maximum(self.max_update_size[i], gparam.max()))
				gparam = T.clip(gparam, -self.clip_range, self.clip_range)
				updates[param] = param - self.learning_rate * gparam

			i+=1

		self.update_fun      = theano.function([indices, label], cost, updates = updates, mode = self.theano_mode)


	def _create_exterior_update_mechanism(self):
		self.create_update_mechanisms()
		self._create_linear_update_mechanism()

	def _select_update_mechanism(self, update_function_name):
		"""
		Choose among these update mechanisms:
			 'hessian' : hessian-free updates,
			  'linear' : linear decay,
			 'clipped' : clipped strategy.
			'exterior' : created gradient_fun,
						alternate_gradient_fun, &
						loss_fun, to power external
						optimizers.
		"""
		update_function_name = update_function_name.lower()
		if update_function_name == 'hessian':
			self.create_update_fun = self._create_hessian_update_mechanism
		elif update_function_name == 'linear':
			self.create_update_fun = self._create_linear_update_mechanism
		elif update_function_name == 'clipped':
			self.create_update_fun = self._create_clipped_update_mechanism
		elif update_function_name == 'adagradclipped' or update_function_name == 'clippedadagrad':
			self.create_update_fun = self._create_clipped_adagrad_update_mechanism
		elif update_function_name == 'exterior':
			self.create_update_fun = self._create_exterior_update_mechanism
		else:
			raise NotImplementedError