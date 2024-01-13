package sliceutil

// Zip returns a slice, where the i-th slice contains the i-th element from each of the argument.
// If all the slices are of the same length, the operation is equivalent to a matrix transpose.
// Excessive elements are discarded.
// Length of the returned slice is equal to the length of the shortest slice passed as an argument.
// Calling Zip again on the result will reverse the process, however the result may not be the same as the original argument.
func Zip[T ~[]E, E any](unzipped ...T) [][]E {
	if unzipped == nil {
		return nil
	}

	if len(unzipped) == 0 {
		return [][]E{}
	}

	cols := len(unzipped)
	rows := len(unzipped[0])
	for _, slice := range unzipped {
		if len(slice) < rows {
			rows = len(slice)
		}
	}

	zipped := make([][]E, rows)
	arr := make([]E, rows*len(unzipped))
	for i := 0; i < rows; i++ {
		row := arr[:cols:cols]
		arr = arr[cols:]
		for j := 0; j < cols; j++ {
			row[j] = unzipped[j][i]
		}
		zipped[i] = row
	}

	return zipped
}
