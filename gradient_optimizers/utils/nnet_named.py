from .nnet import GradientStatus

class NamedGradientStatus(GradientStatus):

    def update_leaf_gradient(self, leaf_name, derivative, lock):
        if lock != None:
            with lock:
                self.set_or_increment_named_param( leaf_name , derivative)
        else:
            self.set_or_increment_named_param( leaf_name , derivative)
        return None

    def update_labeling_error(self, labeling_matrix, labeling_gradient, error, lock):
        if lock != None:
            with lock:
                self.error     += error
                self.batchsize += 1
                self[labeling_matrix] += labeling_gradient
        else:
            self[labeling_matrix] += labeling_gradient
            self.error     += error
            self.batchsize += 1