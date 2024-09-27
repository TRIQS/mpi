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

struct non_mpi_t {
  int a{1};
};

// needs to be in the mpi namespace for ADL to work
namespace mpi {

  // specialize mpi_broadcast for foo
  void mpi_broadcast(non_mpi_t &f, mpi::communicator c = {}, int root = 0) { broadcast(f.a, c, root); }

  // specialize mpi_reduce_in_place for foo
  void mpi_reduce_in_place(non_mpi_t &f, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    if (all) {
      all_reduce_in_place(f.a, c, op);
    } else {
      reduce_in_place(f.a, c, root, false, op);
    }
  }

  // specialize mpi_reduce for foo
  non_mpi_t mpi_reduce(non_mpi_t const &f, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    non_mpi_t res{};
    if (all) {
      res.a = all_reduce(f.a, c, op);
    } else {
      res.a = reduce(f.a, c, root, false, op);
    }
    return (c.rank() == root || all ? res : non_mpi_t{});
  }

} // namespace mpi
