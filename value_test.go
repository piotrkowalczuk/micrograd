package micrograd_test

import (
	"testing"

	mg "github.com/piotrkowalczuk/micrograd"
)

func TestValue_a(t *testing.T) {
	x1 := mg.V(2.0, mg.Label("x1"))
	x2 := mg.V(0.0, mg.Label("x2"))
	// weights
	w1 := mg.V(-3.0, mg.Label("w1"))
	w2 := mg.V(1.0, mg.Label("w2"))
	// bias
	b := mg.V(6.8813735870195432, mg.Label("b"))
	x1w1 := x1.Mul(w1, mg.Label("x1*w1"))
	x2w2 := x2.Mul(w2, mg.Label("x2*w2"))
	x1w1x2w2 := x1w1.Add(x2w2, mg.Label("x1*w1 + x2*w2"))
	n := x1w1x2w2.Add(b, mg.Label("n"))
	o := n.Tanh(mg.Label("o"))

	t.Log("=================")
	t.Log(x1w1x2w2)
	t.Log(n)
	t.Log(o)
	t.Log("=================")
	o.Backward()
	t.Log(x1w1x2w2)
	t.Log(n)
	t.Log(o)
	t.Log("=================")
	n.Backward()
	x1w1x2w2.Backward()
	x1w1.Backward()
	x2w2.Backward()
	t.Log(x1, w1, "----", x2, w2)
	t.Log(x1w1, x2w2)
	t.Log(x1w1x2w2)
	t.Log(n)
	t.Log(o)
}

func TestValue_b(t *testing.T) {
	a := mg.V(3.0, mg.Label("a"))
	b := a.Add(a, mg.Label("b"))
	b.Backward()
	t.Log(a)
	t.Log(b)
}
