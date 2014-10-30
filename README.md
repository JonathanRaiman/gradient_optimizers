Two classes GradientModel, and GradientHFModel, for optimizing gradient
based models (specifically built with indexed parameters in mind (e.g.
for language models))

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

    ivector <indices/>, ivector <labels/>

If this is not the case you can modify them as appropriate.

GradientHFModel
---------------

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