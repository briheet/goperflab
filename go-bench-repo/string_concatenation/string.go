package stringconcatenation

import (
	"fmt"
)

func calculateResult(num int) string {

	var result string
	for i := range num {
		result += fmt.Sprintf("item-%d,", i)
	}

	return result
}
