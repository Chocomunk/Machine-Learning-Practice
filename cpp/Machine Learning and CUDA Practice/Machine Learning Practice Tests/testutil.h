#pragma once

bool vecIsEqual(const int n, float *a, float *b) {
	for (int i = 0; i < n; i++) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}
