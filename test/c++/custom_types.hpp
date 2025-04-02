// Copyright (c) 2022-2024 Simons Foundation
// Copyright (c) 2022 Hugo U.R. Strand
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Hugo U.R. Strand

#pragma once

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <tuple>

// Custom type which is MPI compatible.
struct mpi_t {
  long a{0};
  bool operator==(const mpi_t &) const = default;
  mpi_t operator+(mpi_t x) const {
    x.a += a;
    return x;
  }
};

// Tie the data (to make it MPI compatible).
inline auto tie_data(mpi_t const &x) { return std::tie(x.a); }

// Custom type which is not MPI compatible but has specialized mpi_xxx implementations.
struct non_mpi_t {
  int a{1};
  bool operator==(const non_mpi_t &) const = default;
};

// Specialize mpi_broadcast for non_mpi_t.
void mpi_broadcast(non_mpi_t &x, mpi::communicator c = {}, int root = 0) { broadcast(x.a, c, root); }

// Specialize mpi_reduce_into for non_mpi_t.
void mpi_reduce_into(non_mpi_t const &in, non_mpi_t &out, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
  mpi::reduce_into(in.a, out.a, c, root, all, op);
}
