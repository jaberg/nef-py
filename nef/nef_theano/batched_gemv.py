
from theano import Op

class DoubleBatchedGemv(Op):
    def __init__(self):
        self._attrs = (,)

    def __eq__(self, other):
        return type(self) == type(other) and self._attrs = other._attrs

    def __hash__(self, other):
        return hash((type(self), self._attrs))

    def make_node(self, *inputs):
        inputs  = map(TT.as_tensor_variable, inputs)
        return theano.Apply(self, inputs, [inputs[0].type()])

    def perform(self, node, inputs, outstor):
        x = inputs[0]
        assert len(inputs) == self.N * 4 + 1
        Ustarts = args[1::4]
        Vstarts = args[2::4]
        Us = args[3::4]
        Vs = args[4::4]
        assert len(Vs) == len(starts)

        for st, le, U, V in zip(starts, lens, Us, Vs):
            raise NotImplementedError()


