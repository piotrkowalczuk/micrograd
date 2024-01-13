package micrograd

import (
	"fmt"
	"math"
	"strings"
)

type Value struct {
	value    float64
	grad     float64
	children Values
	label    string
	op       string
	backward func()
}

func V(data float64, options ...Option) *Value {
	opts := opts{}
	for _, o := range options {
		o(&opts)
	}

	return &Value{
		value:    data,
		label:    opts.label,
		backward: func() {},
		grad:     0.0,
	}
}

// String implements fmt.Stringer interface.
func (v *Value) String() string {
	if v == nil {
		return "[ nil ]"
	}
	return fmt.Sprintf("[ %s | %s | %f | %f ]", v.op, v.label, v.value, v.grad)
}

// Float64 returns the value as float64.
func (v *Value) Float64() float64 {
	return v.value
}

// Gradient returns the gradient of the value.
func (v *Value) Gradient() float64 {
	return v.grad
}

// SetValue sets the value.
func (v *Value) SetValue(val float64) {
	v.value = val
}

// SetGradient sets the gradient.
func (v *Value) SetGradient(val float64) {
	v.grad = val
}

// Add adds two values.
func (v *Value) Add(w *Value, options ...Option) *Value {
	opts := opts{}
	for _, o := range options {
		o(&opts)
	}

	o := &Value{
		value:    v.value + w.value,
		label:    opts.label,
		children: Values{v, w},
		op:       "+",
	}
	o.backward = func() {
		// fmt.Println("backward-add", o.grad)
		v.grad += o.grad
		w.grad += o.grad
	}

	return o
}

// Sub subtracts two values.
func (v *Value) Sub(w *Value, options ...Option) *Value {
	return v.Add(w.Neg(), options...)
}

// Mul multiplies two values.
func (v *Value) Mul(w *Value, options ...Option) *Value {
	opts := opts{}
	for _, o := range options {
		o(&opts)
	}

	o := &Value{
		value:    v.value * w.value,
		label:    opts.label,
		children: Values{v, w},
		op:       "*",
	}
	o.backward = func() {
		// fmt.Println("backward-mul", w.value*o.grad)
		v.grad += w.value * o.grad
		w.grad += v.value * o.grad
	}

	return o
}

// Neg negates the value.
func (v *Value) Neg() *Value {
	return v.Mul(V(-1))
}

// Tanh computes the hyperbolic tangent of the value.
func (v *Value) Tanh(options ...Option) *Value {
	opts := opts{}
	for _, o := range options {
		o(&opts)
	}

	o := &Value{
		value: math.Tanh(v.value),
		// value:    (math.Exp(2*v.value) - 1) / (math.Exp(2*v.value) + 1),
		label:    opts.label,
		children: Values{v},
		op:       "tanh",
	}
	o.backward = func() {
		v.grad += (1 - math.Pow(o.value, 2)) * o.grad
	}

	return o
}

// Pow computes the power of the value.
func (v *Value) Pow(pow float64, options ...Option) *Value {
	opts := opts{}
	for _, o := range options {
		o(&opts)
	}

	o := &Value{
		value:    math.Pow(v.value, pow),
		label:    opts.label,
		children: Values{v},
		op:       "pow2",
	}
	o.backward = func() {
		v.grad += (pow * math.Pow(v.value, pow-1)) * o.grad
	}

	return o
}

//var count int64
//
//// Backward computes the gradients and propagates it up.
//func (v *Value) Backward() {
//	v.grad = 1.0
//	visited := make(map[*Value]struct{})
//	var traverse func(*Value)
//	traverse = func(v *Value) {
//		if _, ok := visited[v]; ok {
//			return
//		}
//
//		visited[v] = struct{}{}
//		v.backward()
//		atomic.AddInt64(&count, 1)
//		for _, c := range v.children {
//			traverse(c)
//		}
//	}
//
//	traverse(v)
//	atomic.StoreInt64(&count, 0)
//}

func (v *Value) Backward() {
	var topo []*Value
	visited := map[*Value]bool{}

	topo = buildTopo(v, topo, visited)

	v.grad = 1.0

	for i := len(topo) - 1; i >= 0; i-- {
		if len(topo[i].children) != 0 {
			topo[i].backward()
		}
	}
}

func buildTopo(v *Value, topo []*Value, visited map[*Value]bool) []*Value {
	if !visited[v] {
		visited[v] = true
		for _, prev := range v.children {
			topo = buildTopo(prev, topo, visited)
		}
		topo = append(topo, v)
	}
	return topo
}

type Values []*Value

func (v Values) String() string {
	var b strings.Builder
	for _, vv := range v {
		b.WriteString(vv.String())
		b.WriteRune('\n')
	}

	return b.String()
}

// Option is a function that modifies the value.
type Option func(*opts)

type opts struct {
	label string
}

// Label sets the label of the value.
func Label(label string) func(*opts) {
	return func(o *opts) {
		o.label = label
	}
}
