"""
Our network class. A network is essentially a sequence of layers.
"""
class Network():
  def __init__(self, layers):
    # layers should be an ordered list of Layer objects
    self.layers = layers

  # pass a single datum prediction through the net
  def predict(self, datum):
    new_datum = datum
    for layer in self.layers:
      new_datum = layer.predict(new_datum)
    return new_datum

  # pass prediction of all data through the net
  def predictAll(self, data):
    predictions = []
    for datum in data:
      predictions.append(self.predict(datum))
    return predictions
