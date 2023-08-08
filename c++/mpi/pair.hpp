// Copyright (c) 2021 Simons Foundation
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
// Authors: Nils Wentzell

#pragma once
#include "mpi.hpp"
#include <tuple>

namespace mpi {

  // ---------------- broadcast ---------------------

  template <typename T1, typename T2> void mpi_broadcast(std::pair<T1, T2> &p, communicator c = {}, int root = 0) {
    broadcast(p.first, c, root);
    broadcast(p.second, c, root);
  }

  // ---------------- reduce ---------------------

  template <typename T1, typename T2>
  auto mpi_reduce(std::pair<T1, T2> const &p, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    return std::make_pair(reduce(p.first, c, root, all, op), reduce(p.second, c, root, all, op));
  }

} // namespace mpi
