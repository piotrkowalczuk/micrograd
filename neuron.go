package micrograd

import (
	"math/rand"
	"strings"
)

type Neuron struct {
	weight Values
	bias   *Value
}

// N instantiates new neuron with random weights and bias.
func N(inputSize int) *Neuron {
	n := &Neuron{
		weight: make(Values, inputSize),
		bias:   V(rand.Float64()*2 - 1),
	}
	for i := range n.weight {
		n.weight[i] = V(rand.Float64()*2 - 1)
	}

	return n
}

// Forward calculates activation of the neuron based on given inputs.
func (n *Neuron) Forward(inputs ...*Value) *Value {
	activation := n.bias
	// for _, val := range sliceutil.Zip(inputs, n.weight) {
	for i := 0; i < len(inputs); i++ {
		activation = activation.Add(n.weight[i].Mul(inputs[i]))
	}

	return activation.Tanh()
}

// Parameters returns all parameters of the neuron.
func (n *Neuron) Parameters() Values {
	parameters := make(Values, len(n.weight)+1)
	copy(parameters, n.weight)
	parameters[len(parameters)-1] = n.bias

	return parameters
}

// String implements fmt.Stringer interface.
func (n *Neuron) String() string {
	var b strings.Builder
	for _, nn := range n.weight {
		b.WriteString(nn.String())
		b.WriteRune('\t')
	}
	return b.String()
}

// Weight returns weight at given index.
func (n *Neuron) Weight(index int) *Value {
	return n.weight[index]
}

// Bias returns bias.
func (n *Neuron) Bias() *Value {
	return n.bias
}
