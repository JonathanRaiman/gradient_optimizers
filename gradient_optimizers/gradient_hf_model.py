import numpy as np, theano, theano.tensor as T, time
from numpy import zeros_like, ones_like
from .utils.nnet import batch_repeat, multi_grad, theano_unique
from collections import OrderedDict

class GradientHFModel(object):
	"""
	Implements an symbolic one step of hessian-free [1]
	optimization that approximates the curvature,
	requires a _compute_cost method that takes an example
	as input or a _compute_cost_gradients that returns
	gradients for each example provided.

	Model should have a params property containing symbolic
	theano variables.

	[1] James Martens, ``Deep learning via Hessian-free optimization", ICML 2010

	Make sure the following parameters are not tampered with:

		self._additional_params

		self._num_updates

	"""

	callback = lambda * _: None
	epsilon = np.float32(1e-12)
	outlier_level = 1

	# Statistics storage and creation:
	# --------------------------------

	def _create_g_h_v_bar_storage(self, param):
		"""
		Creates specialized shared variables for variance and hessian
		estimates for optimal learning rates.

		Todo:
		- add initial samples initialization
		"""
		self._additional_params[param]["gbar"]  = theano.shared(zeros_like(param.get_value(borrow=True)).astype(param.dtype), name=('%s_gbar' % (param.name)))
		self._additional_params[param]["vbar"]  = theano.shared((ones_like(param.get_value(borrow=True)) * self.epsilon).astype(param.dtype), name=('%s_vbar' % (param.name)))
		self._additional_params[param]["hbar"]  = theano.shared(zeros_like(param.get_value(borrow=True)).astype(param.dtype), name=('%s_hbar' % (param.name)))
		self._additional_params[param]["vpart"] = theano.shared(self._additional_params[param]["gbar"].get_value(borrow=True) ** 2 / self._additional_params[param]["vbar"].get_value(borrow=True), name=('%s_vpart' % (param.name)))
		self._additional_params[param]["vhbar"] = theano.shared(ones_like(param.get_value(borrow=True)).astype(param.dtype) * self.epsilon, name=('%s_vhbar' % (param.name)))
		self._additional_params[param]["hpart"] = theano.shared( ((self.epsilon) / (self._additional_params[param]["vhbar"].get_value(borrow = True) + self.epsilon)  ).astype(param.dtype), name=('%s_hpart' % (param.name)))

	def _create_tau_storage(self, param):
		self._additional_params[param]["taus"] = theano.shared(((ones_like(param.get_value(borrow=True))+self.epsilon) * 2).astype(param.dtype), name=('%s_taus' % (param.name)))

	def _create_diaghessian_storage(self, param):
		self._additional_params[param]["last_diaghessians"] = theano.shared(batch_repeat(zeros_like(param.get_value(borrow=True)).astype(param.dtype), self.batch_size), name=('%s_last_diaghessians' % (param.name)))

	def _create_last_gradients_storage(self, param):
		self._additional_params[param]["last_gradients"] = theano.shared(batch_repeat(zeros_like(param.get_value(borrow=True)).astype(param.dtype), self.batch_size), name=('%s_last_gradients' % (param.name)))

	def _create_hessian_update_mechanism(self):

		# create hessian theano variables:
		self.create_shared_variables_hessian_variables()

		# symbolic examples:
		examples = []
		example_indices = []
		examples_tuples = []
		for i in range(self.batch_size):
			index, label = T.ivectors(['indices', 'labels'])
			examples_tuples.append((index, label))
			examples.append(index)
			example_indices.append(index)
			examples.append(label)

		indices = theano_unique(T.concatenate(example_indices))

		t0 = time.time()

		print("1/2 Calculating gradients...")

		hf_updates, costs = self.symbolic_one_step(examples_tuples, indices)

		t1 = time.time()
		print("Took %.2fs to compute gradients\n2/2 Compiling hessian updates..." % (t1 - t0))

		self.update_fun = theano.function(examples, sum(costs), updates = hf_updates, mode = self.theano_mode)

		t2 = time.time()
		print("Took %.2fs to compile hessian updates" % (t2 - t1))

	def create_shared_variables_hessian_variables(self):
		
		self._additional_params = {}
		self._num_updates = theano.shared(0)
		for param in self.params:
			self._additional_params[param] = {}

			# for gradients:
			self._create_last_gradients_storage(param)
			# for outliers:
			self._create_tau_storage(param)
			# for hessian approximation:
			self._create_diaghessian_storage(param)
			# for statistics:
			self._create_g_h_v_bar_storage(param)

	# Statistics Computation:
	# -----------------------

	def update_gradients(self, updates, params, alt_params, alt_params_hash, indices = None, gparams = None, costs = None):
		"""
		For a given parameter, do the following:

		Get an approximation for the hessian diagonal term
		by comparing the current gradient, with an alternate
		obtained by incrementing the parameters by the
		best estimate for the curvature.

		"""
		# get the original gradient:
		if gparams is None:
			if costs is not None:
				gparams = multi_grad(costs, params)
			else:
				raise Exception("Provide either a gradient or a cost.")
		# rerun the gradient with new values:
		alt_gparams = theano.clone(gparams, replace = alt_params_hash)

		def multi_batch_update_with_indices(updates, pp, newpp, inds):
			if indices is not None:
				updates[pp] = T.set_subtensor(pp[0:self.batch_size, inds], newpp)
			else:
				updates[pp] = newpp

		for param, gparam, alt_param, alt_gparam in zip(params, gparams, alt_params, alt_gparams):
			p = self._additional_params[param]
			if hasattr(self, 'index_params') and param in self.index_params and indices is not None:
				if self.batch_size > 1:
					multi_batch_update_with_indices(updates, p["last_gradients"], gparam[0:self.batch_size, indices], indices)
					multi_batch_update_with_indices(updates, p["last_diaghessians"], abs((gparam[0:self.batch_size, indices] - alt_gparam[0:self.batch_size, indices]) / (p["gbar"][indices] + self.epsilon)), indices)
				else:
					if indices is not None:
						updates[p["last_gradients"]] = T.set_subtensor(p["last_gradients"][0, indices], gparam)
						updates[p["last_diaghessians"]] = T.set_subtensor(p["last_diaghessians"][0], abs((gparam - alt_gparam) / (p["gbar"][indices] + self.epsilon)))
					else:
						updates[p["last_gradients"]] = T.set_subtensor(p["last_gradients"][0, indices], gparam)
						updates[p["last_diaghessians"]] = T.set_subtensor(p["last_diaghessians"][0, indices], abs((gparam - alt_gparam) / (p["gbar"][indices] + self.epsilon)))
			else:
				if self.batch_size > 1:
					updates[p["last_gradients"]] = gparam
					updates[p["last_diaghessians"]] = abs((gparam - alt_gparam) / (p["gbar"] + self.epsilon))
				else:
					updates[p["last_gradients"]] = T.set_subtensor(p["last_gradients"][0], gparam)
					updates[p["last_diaghessians"]] = T.set_subtensor(p["last_diaghessians"][0], abs((gparam - alt_gparam) / (p["gbar"] + self.epsilon)))			

		return updates

	def _detectOutliers(self, updates, param, indices = None):
		""" Binary vector for which dimension saw an outlier gradient. """
		p = self._additional_params[param]
		var = (self._additional_params[param]["vbar"] - self._additional_params[param]["gbar"] ** 2) / self.batch_size
		if indices is not None:
			return ((((updates[p["last_gradients"]].mean(axis=0) - self._additional_params[param]["gbar"]) ** 2) > ((self.outlier_level ** 2) * var))[indices]).nonzero()
		else:
			return (((updates[p["last_gradients"]].mean(axis=0) - self._additional_params[param]["gbar"]) ** 2) > ((self.outlier_level ** 2) * var)).nonzero()

	def symbolic_one_step(self, examples, indices = None, statistics = True, update_parameters = True):
		"""
		One step of optimization
		------------------------

		Takes as input a set of symbolic examples
		and creates a symbolic step of hessian hessian-free
		optimization.

		Inputs
		------

					  list<examples> : a list of examples compatible with
									   the model's `_compute_cost` function.
		  indices<ivector>(optional) : indices for index_params
									   to update.

		Outputs
		-------

		(updates, costs) : OrderedDict of parameter updates,
						   and symbolic list of costs for each
						   example.

		"""

		updates    = OrderedDict()

		alt_params = []
		alt_params_hash = {}

		# create the alt params using an fd direction:
		for param in self.params:
			p            = self._additional_params[param]
			if hasattr(self, 'index_params') and param in self.index_params:
				fd_direction = p["gbar"][indices] + self.epsilon
				alt_param    = T.inc_subtensor(param[indices], fd_direction)
			else:
				fd_direction = p["gbar"] + self.epsilon
				alt_param    = param     + fd_direction
			alt_params.append(alt_param)
			alt_params_hash[param] = alt_param

		if hasattr(self, '_compute_cost_gradients'):
			gparams, costs = self._compute_cost_gradients(examples)
			updates   = self.update_gradients(updates, self.params, alt_params, alt_params_hash, indices, gparams = gparams)
		else:
			costs     = self._compute_cost(examples)
			updates   = self.update_gradients(updates, self.params, alt_params, alt_params_hash, indices, costs = costs)

		if statistics:
			self.symbolic_compute_statistics(updates, indices)

		if update_parameters:
			self.symbolic_update_parameters(updates, indices)

		return (updates, costs)

	def symbolic_compute_statistics(self, updates = OrderedDict(), indices = None):
		for param in self.params:
			if hasattr(self, 'index_params') and param in self.index_params:
				updates = self._computeStatistics(updates, param, indices)
			else:
				updates = self._computeStatistics(updates, param)
		return updates

	def symbolic_update_parameters(self, updates = OrderedDict(), indices = None):
		for param in self.params:
			if hasattr(self, 'index_params') and param in self.index_params:
				updates = self._update_parameter(updates, param, indices)
			else:
				updates = self._update_parameter(updates, param)

		# # here we can the theano op for this function.
		updates[self._num_updates] = self._num_updates + 1

		return updates


	def _update_parameter(self, updates, param, indices = None):
		p = self._additional_params[param]

		# vsgd (Schaul, Zhang & LeCun 2012)
		#learning_rate = updates[p["vpart"]] / (updates[p["hbar"]] + epsilon)

		# vSGD (Schaul Lecun 2013) w/. finite difference approximation
		if indices is not None:
			learning_rate = updates[p["vpart"]][indices] * updates[p["hpart"]][indices]
			last_gradient = updates[p["last_gradients"]][0:self.batch_size, indices].mean(axis = 0)
			updates[param] = T.inc_subtensor(param[indices], - learning_rate * last_gradient)
		else:
			learning_rate = updates[p["vpart"]] * updates[p["hpart"]]
			last_gradient = updates[p["last_gradients"]].mean(axis=0)
			updates[param] = param - learning_rate * last_gradient

		return updates

	def _computeStatistics(self, updates, param, indices = None, use_saved_gradients = False):
		"""

		For a specific parameter computes a series of updates
		to each statistical param and returns an OrderedDict
		with those changes.

		"""
		def update_with_indices(updates, pp, newpp, inds):
			if indices is not None:
				updates[pp] = T.set_subtensor(pp[inds], newpp)
			else:
				updates[pp] = newpp

		p = self._additional_params[param]

		if indices is not None:
			if use_saved_gradients:
				grads = p["last_gradients"][0:self.batch_size, indices]
				hs    = p["last_diaghessians"][0:self.batch_size, indices].mean(axis=0)
				sq_hs = (p["last_diaghessians"][0:self.batch_size, indices] ** 2).mean(axis=0)
			else:
				grads = updates[p["last_gradients"]][0:self.batch_size, indices]
				hs    = updates[p["last_diaghessians"]][0:self.batch_size, indices].mean(axis=0)
				sq_hs = (updates[p["last_diaghessians"]][0:self.batch_size, indices] ** 2).mean(axis=0)
			gbar  = p["gbar"][indices]
			vbar  = p["vbar"][indices]
			hbar  = p["hbar"][indices]
			vhbar = p["vhbar"][indices]
		else:
			if use_saved_gradients:
				grads = p["last_gradients"]
				hs    = p["last_diaghessians"].mean(axis=0)
				sq_hs = (p["last_diaghessians"] ** 2).mean(axis=0)
			else:
				grads = updates[p["last_gradients"]]
				hs    = updates[p["last_diaghessians"]].mean(axis=0)
				sq_hs = (updates[p["last_diaghessians"]] ** 2).mean(axis=0)
			gbar  = p["gbar"]
			vbar  = p["vbar"]
			hbar  = p["hbar"]
			vhbar = p["vhbar"]

		# slow down updates if the last sample was an outlier
		if self.outlier_level is not None:
			if param.ndim > 0:
				new_taus = T.inc_subtensor(p["taus"][self._detectOutliers(updates, param, indices)], 1)
				updates[p["taus"]] = new_taus
			else:
				new_taus = p["taus"]
		
		# update statistics
		if self.outlier_level is not None:
			fract = (1. / new_taus[indices]) if indices is not None else (1. / new_taus)
		else:
			fract = np.float32(0.5)

		opposite_fract = (1. - fract)

		# take running mean of gradients:
		new_gbar = gbar * opposite_fract + fract * grads.mean(axis = 0)
		update_with_indices(updates, p["gbar"], new_gbar, indices)

		# # take running mean of squared gradients:
		new_vbar = vbar * opposite_fract + fract * (grads **2).mean(axis=0) + self.epsilon
		update_with_indices(updates, p["vbar"], new_vbar, indices)

		# # take running estimate of hessian diagonal:
		new_hbar = hbar * opposite_fract + fract * hs
		update_with_indices(updates, p["hbar"], new_hbar, indices)
		
		# # update time constants based on the variance-part of the learning rate:
		if self.batch_size > 1:  
			new_vpart = new_gbar ** 2 / (1. / self.batch_size * new_vbar + 
									   (self.batch_size - 1.) / self.batch_size * (new_gbar ** 2))
		else:
			new_vpart = new_gbar ** 2 / new_vbar

		update_with_indices(updates, p["vpart"], new_vpart, indices)
		
		if self.outlier_level is not None:
			if indices is not None:
				new_taus = new_taus[indices] * ( 1.  - new_vpart) + 1 + self.epsilon
			else:
				new_taus = new_taus * ( 1.  - new_vpart) + 1 + self.epsilon
			update_with_indices(updates, p["taus"], new_taus, indices)
		
		new_vhbar = vhbar * (1. - fract) + fract * sq_hs
		update_with_indices(updates, p["vhbar"], new_vhbar, indices)

		new_hpart = (new_hbar + self.epsilon) / (new_vhbar + self.epsilon)
		update_with_indices(updates, p["hpart"], new_hpart, indices)

		return updates