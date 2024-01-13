package micrograd

import (
	"strings"
)

// MultiLayerPerceptron is a neural network with multiple layers.
type MultiLayerPerceptron struct {
	layers []*Layer
}

// MLP creates new multi layer perceptron.
func MLP(inputSize int, outputSizes ...int) *MultiLayerPerceptron {
	mlp := &MultiLayerPerceptron{layers: make([]*Layer, len(outputSizes))}
	for i := range mlp.layers {
		mlp.layers[i] = L(inputSize, outputSizes[i])
		inputSize = outputSizes[i]
	}

	return mlp
}

// Forward calculates activation of the neuron based on given inputs.
func (mlp *MultiLayerPerceptron) Forward(input ...*Value) []*Value {
	for _, l := range mlp.layers {
		input = l.Forward(input...)
	}

	return input
}

// Parameters returns all parameters of the neuron.
func (mlp *MultiLayerPerceptron) Parameters() Values {
	var length int
	for _, l := range mlp.layers {
		length += len(l.neurons) * (len(l.neurons[0].weight) + 1)
	}

	parameters := make(Values, 0, length)
	for _, l := range mlp.layers {
		parameters = append(parameters, l.Parameters()...)
	}

	return parameters
}

// String implements fmt.Stringer interface.
func (mlp *MultiLayerPerceptron) String() string {
	var b strings.Builder
	for _, l := range mlp.layers {
		b.WriteString(l.String())
		b.WriteRune('\n')
	}
	return b.String()
}

// Layer returns layer at given index.
func (mlp *MultiLayerPerceptron) Layer(index int) *Layer {
	return mlp.layers[index]
}
