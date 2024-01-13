package sliceutil_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/piotrkowalczuk/micrograd/internal/sliceutil"
)

func TestZip_matrix(t *testing.T) {
	cases := map[string]struct {
		given     [][]int
		exp, exp2 [][]int
	}{
		"nil": {
			given: nil,
			exp:   nil,
			exp2:  nil,
		},
		"empty": {
			given: [][]int{},
			exp:   [][]int{},
			exp2:  [][]int{},
		},
		"one": {
			given: [][]int{
				{1, 2, 3},
			},
			exp: [][]int{
				{1},
				{2},
				{3},
			},
			exp2: [][]int{
				{1, 2, 3},
			},
		},
		"transpose": {
			given: [][]int{
				{1, 3, 5},
				{2, 4, 6},
			},
			exp: [][]int{
				{1, 2},
				{3, 4},
				{5, 6},
			},
			exp2: [][]int{
				{1, 3, 5},
				{2, 4, 6},
			},
		},
		"uneven": {
			given: [][]int{
				{1, 2, 3},
				{4, 5},
			},
			exp: [][]int{
				{1, 4},
				{2, 5},
			},
			exp2: [][]int{
				{1, 2},
				{4, 5},
			},
		},
	}
	for hint, c := range cases {
		t.Run(hint, func(t *testing.T) {
			got := sliceutil.Zip[[]int](c.given...)
			if cmp.Diff(c.exp, got) != "" {
				t.Fatalf("unexpected result, diff: %s", cmp.Diff(c.exp, got))
			}

			got = sliceutil.Zip[[]int](got...)
			if cmp.Diff(c.exp2, got) != "" {
				t.Fatalf("unexpected second result, diff: %s", cmp.Diff(c.exp2, got))
			}
		})
	}
}

func TestZip_tensor(t *testing.T) {
	cases := map[string]struct {
		given     [][][]int
		exp, exp2 [][][]int
	}{
		"nil": {
			given: nil,
			exp:   nil,
			exp2:  nil,
		},
		"empty": {
			given: [][][]int{},
			exp:   [][][]int{},
			exp2:  [][][]int{},
		},
		"uneven": {
			given: [][][]int{
				{{1, 2}, {2, 3}, {4, 5}},
				{{6, 7}, {8, 9}},
			},
			exp: [][][]int{
				{{1, 2}, {6, 7}},
				{{2, 3}, {8, 9}},
			},
			exp2: [][][]int{
				{{1, 2}, {2, 3}},
				{{6, 7}, {8, 9}},
			},
		},
	}
	for hint, c := range cases {
		t.Run(hint, func(t *testing.T) {
			got := sliceutil.Zip[[][]int](c.given...)
			if cmp.Diff(c.exp, got) != "" {
				t.Fatalf("unexpected result, diff: %s", cmp.Diff(c.exp, got))
			}

			got = sliceutil.Zip[[][]int](got...)
			if cmp.Diff(c.exp2, got) != "" {
				t.Fatalf("unexpected second result, diff: %s", cmp.Diff(c.exp2, got))
			}
		})
	}
}

func BenchmarkZip(b *testing.B) {
	data := [][]int{
		{1, 3, 5},
		{2, 4, 6},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sliceutil.Zip[[]int](data...)
	}
}
