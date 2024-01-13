package micrograd_test

import (
	"testing"

	"github.com/piotrkowalczuk/micrograd"
)

func TestMultiLayerPerceptron(t *testing.T) {
	cases := []struct {
		input  []*micrograd.Value
		output float64
	}{
		{
			input:  []*micrograd.Value{micrograd.V(2), micrograd.V(3), micrograd.V(-1)},
			output: 1.0,
		},
		{
			input:  []*micrograd.Value{micrograd.V(3), micrograd.V(-1), micrograd.V(0.5)},
			output: -1.0,
		},
		{
			input:  []*micrograd.Value{micrograd.V(0.5), micrograd.V(1.0), micrograd.V(1.0)},
			output: -1.0,
		},
		{
			input:  []*micrograd.Value{micrograd.V(1), micrograd.V(1), micrograd.V(-1)},
			output: 1.0,
		},
	}

	mlp := micrograd.MLP(3, 4, 4, 1)
	var (
		finalLoss        *micrograd.Value
		finalPredictions []*micrograd.Value
	)
	for i := 0; i < 100; i++ {
		predictions := make([]*micrograd.Value, 0, len(cases))
		for _, c := range cases {
			predictions = append(predictions, mlp.Forward(c.input...)[0])
		}

		loss := micrograd.V(0.0, micrograd.Label("loss"))
		for i := range cases {
			sub := predictions[i].Sub(micrograd.V(cases[i].output))
			pow := sub.Pow(2)
			loss = loss.Add(pow)
		}
		for _, p := range mlp.Parameters() {
			p.SetGradient(0)
		}
		loss.Backward()

		// fmt.Println("========================================")
		// fmt.Println("LOSS", loss.String())
		// fmt.Println("GRAD", mlp.Layer(0).Neuron(0).Weight(0).Gradient())
		// fmt.Println("VALS", mlp.Layer(0).Neuron(0).Weight(0).Float64())
		// fmt.Println("PRED", predictions)

		for _, p := range mlp.Parameters() {
			p.SetValue(p.Float64() + -0.05*p.Gradient())
		}
		finalLoss = loss
		finalPredictions = predictions
	}

	if finalLoss.Float64() > 0.05 {
		t.Errorf("expected loss to be less than 0.05, final loss %f", finalLoss.Float64())
	}

	for i, prediction := range finalPredictions {
		if prediction.Float64() < 0.9*cases[i].output && prediction.Float64() < 1.1*cases[i].output {
			t.Errorf("expected prediction to be close to %f, got %f", cases[i].output, prediction.Float64())
		}
	}

	t.Log("final loss", finalLoss.Float64())
	t.Log("final predictions", finalPredictions)
}
