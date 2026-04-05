package stringconcatenation

import (
	"testing"
)

func TestCalculateResult(t *testing.T) {

	t.Run("basic test", func(t *testing.T) {

		num := 2
		got := calculateResult(num)
		want := "item-0,item-1,"

		if got != want {
			t.Errorf("got %s, want %s", got, want)
		}
	})
}

func BenchmarkCalculateResult(b *testing.B) {

	num := 10_000
	for b.Loop() {
		calculateResult(num)
	}
}
