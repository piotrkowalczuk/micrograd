package micrograd

import "strings"

// Layer is a collection of neurons.
type Layer struct {
	neurons []*Neuron
}

// L instantiates new layer with random weights and bias.
func L(inputSize, outputSize int) *Layer {
	l := &Layer{
		neurons: make([]*Neuron, outputSize),
	}

	for i := range l.neurons {
		l.neurons[i] = N(inputSize)
	}

	return l
}

// Forward calculates activation of the neuron based on given inputs.
// It returns
func (l *Layer) Forward(input ...*Value) Values {
	outputs := make(Values, len(l.neurons))
	for i, n := range l.neurons {
		outputs[i] = n.Forward(input...)
	}

	return outputs
}

// Parameters returns all parameters of all neurons in the layer.
func (l *Layer) Parameters() Values {
	var length int
	for _, n := range l.neurons {
		length += len(n.weight) + 1
	}

	parameters := make(Values, 0, length)
	for _, n := range l.neurons {
		parameters = append(parameters, n.Parameters()...)
	}

	return parameters
}

// String implements fmt.Stringer interface.
func (l *Layer) String() string {
	var b strings.Builder
	b.WriteString("====================================\n")
	for _, n := range l.neurons {
		b.WriteString(n.String())
		b.WriteRune('\n')
	}
	b.WriteString("====================================\n")
	return b.String()
}

// Neuron returns neuron at given index.
func (l *Layer) Neuron(index int) *Neuron {
	return l.neurons[index]
}
