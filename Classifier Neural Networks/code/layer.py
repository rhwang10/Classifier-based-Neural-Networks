from numpy import zeros, dot, array
from sklearn.utils import resample


class Layer():
  """
  usage for constructor example:
  Layer(DecisionTreeClassifier, {'max_depth': 5}, X, y, 5)
  """
  def __init__(self, classifier, classifier_params, data, labels, num_nodes):
    self.nodes = []
    self.weights = []
    # make as many nodes in the layer as we want
    # make a parallel array of weights, all initialized to 1
    for i in range(num_nodes):
      self.nodes.append(classifier(**classifier_params))
    for i in range(len(data[0])):
      self.weights.append(1.0)

    for node in self.nodes:
      if len(self.nodes) > 1:
        X, y = resample(data, labels, n_samples=len(data), random_state=42)
      else:
        # use the full data for logistic regression
        X, y = data, labels
      node.fit(X, y)

  # runs the vector (either from the last layer, or the original feature vector)
  # through each tree in the layer, multiplying their weight matrix by the
  # vector, and then compiles each decision into a vector to pass on
  def predict(self, datum):
    vector = []
    for tree in self.nodes:
      product = array(datum) * array(self.weights)
      vector.append(tree.predict(product.reshape(1, -1))[0])
    return array(vector)

  # probably will only used during generation of the net
  def predictAll(self, data):
    layer_data = []
    for datum in data:
      layer_data.append(self.predict(datum))
    return array(layer_data)
