// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides an MPI broadcast, reduce, scatter and gather for std::vector.
 */

#pragma once

#include "./mpi.hpp"
#include "./ranges.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for a std::arr.
   *
   * @details It simply calls mpi::broadcast_range with the input array.
   *
   * @tparam T Value type of the array.
   * @tparam N Size of the array.
   * @param arr std::array to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T, std::size_t N> void mpi_broadcast(std::array<T, N> &arr, communicator c = {}, int root = 0) { broadcast_range(arr, c, root); }

  /**
   * @brief Implementation of an in-place MPI reduce for a std::array.
   *
   * @details It simply calls mpi::reduce_in_place_range with the given input array.
   *
   * @tparam T Value type of the array.
   * @tparam N Size of the array.
   * @param arr std::array to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T, std::size_t N>
  void mpi_reduce_in_place(std::array<T, N> &arr, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    reduce_in_place_range(arr, c, root, all, op);
  }

  /**
   * @brief Implementation of an MPI reduce for a std::array.
   *
   * @details It simply calls mpi::reduce_range with the given input array and an empty array of the same size.
   *
   * @tparam T Value type of the array.
   * @tparam N Size of the array.
   * @param arr std::array to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return std::array containing the result of each individual reduction.
   */
  template <typename T, std::size_t N>
  std::array<regular_t<T>, N> mpi_reduce(std::array<T, N> const &arr, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    std::array<regular_t<T>, N> res{};
    reduce_range(arr, res, c, root, all, op);
    return res;
  }

  /** @} */

} // namespace mpi
