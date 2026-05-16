// Consolidated hash and equality functors for Determinant<N>
#ifndef CDFCI_HASH_H
#define CDFCI_HASH_H

#include "determinant.h"
#include "lib/robin_hood/robin_hood.h"

template<int N>
struct DeterminantHash {
    size_t operator()(const Determinant<N>& det) const {
        throw std::invalid_argument("The DeterminantHash does not support N > 4.");
        return 0;
    }
};

template<>
struct DeterminantHash<1> {
    size_t operator()(const Determinant<1>& det) const { return det.repr[0]; }
};

template<>
struct DeterminantHash<2> {
    size_t operator()(const Determinant<2>& det) const {
        return det.repr[0] * 2038076783ull + det.repr[1] * 179426549ull;
    }
};

template<>
struct DeterminantHash<3> {
    size_t operator()(const Determinant<3>& det) const {
        return det.repr[0] * 2038076783ull + det.repr[1] * 179426549ull + det.repr[2] * 500002577ull;
    }
};

template<>
struct DeterminantHash<4> {
    size_t operator()(const Determinant<4>& det) const {
        return det.repr[0] * 2038076783ull + det.repr[1] * 179426549ull
             + det.repr[2] * 500002577ull + det.repr[3] * 255477023ull;
    }
};

template<int N>
struct DeterminantHashRobinhood {
    size_t operator()(const Determinant<N>& det) const {
        DeterminantHash<N> hash1;
        return robin_hood::hash_int(hash1(det));
    }
};

template<int N>
struct DeterminantEqual {
    bool operator()(const Determinant<N>& det1, const Determinant<N>& det2) const {
        for (int i = 0; i < N; ++i) {
            if (det1.repr[i] != det2.repr[i]) return false;
        }
        return true;
    }
};

#endif
