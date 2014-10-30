from collections import OrderedDict
import numpy as np, theano, theano.tensor as T

def symbolic_variable_for_dimension(dimension):
    """
    Obtain the right symbolic variable for a dimension of element.
    """
    if  dimension  == 0:
        return T.scalar()
    elif dimension == 1:
        return T.vector()
    elif dimension == 2:
        return T.matrix()
    elif dimension == 3:
        return T.tensor3()
    elif dimension == 4:
        return T.tensor4()
    else:
        raise AssertionError("Should have between 1 and 4 dimensions.")

class UniqueOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.unique(x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        pass

    def R_op(self, inputs, eval_points):
        pass

theano_unique = UniqueOp()

def batch_repeat(elem, batchsize):
    return elem.reshape([1] + list(elem.shape)).repeat(batchsize, 0)

def multi_grad(costs, params):
    """
    Computes the gradient for several different costs
    separately and provides a rank+1 parameter gradient
    for each gradient with dimension 0 corresponding to
    a different cost.

    """
    all_grads = []
    for param in params:
        if len(costs) > 1:
            param_grads = []
            for cost in costs:
                gparam = T.grad(cost, param)
                param_grads.append(gparam)
            all_grads.append(T.stacklists(param_grads))
        else:
            gparam = T.grad(costs[0], param)
            all_grads.append(gparam)
        
    return all_grads

class GradientStatus(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error         = 0.0
        self.batchsize     = 0
        self._named_param = {}

    @property
    def parameters(self):
        params = set()
        params.update(self.keys())
        params.update(self._named_param.keys())
        return params

    def get_named_param(self, name):
        return self._named_param[name]

    def set_or_increment_named_param(self, name, setvalue):
        if self._named_param.get(name, None) != None:
            self._named_param[name] += setvalue
        else:
            self._named_param[name] = setvalue

    def increment_named_param(self, name, increment):
        self._named_param[name] += increment

    def set_named_param(self,name, value):
        self._named_param[name] = value

    def add_symbolic_variables(self, params):
        for param in params:
            self[param] = np.zeros_like(param.get_value(borrow=True))