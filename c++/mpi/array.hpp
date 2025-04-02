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
 * @brief Provides an MPI broadcast and reduce for `std::array`.
 */

#pragma once

#include "./communicator.hpp"
#include "./ranges.hpp"
#include "./utils.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for a `std::array`.
   *
   * @details It calls mpi::broadcast_range with the given array.
   *
   * @tparam T Value type of the array.
   * @tparam N Size of the array.
   * @param arr `std::array` to broadcast (into).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T, std::size_t N> void mpi_broadcast(std::array<T, N> &arr, communicator c = {}, int root = 0) { broadcast_range(arr, c, root); }

  /**
   * @brief Implementation of an MPI reduce for a `std::array`.
   *
   * @details It constructs the output array with its value type equal to the return type of `reduce(std::declval<T>())`
   * and calls mpi::reduce_range with the input and constructed output array.
   *
   * Note that the output array will always have the same size as the input array, no matter if the rank receives the
   * reduced data or not.
   *
   * @tparam T Value type of the array.
   * @tparam N Size of the array.
   * @param arr `std::array` to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return `std::array` containing the result of the reduction.
   */
  template <typename T, std::size_t N>
  auto mpi_reduce(std::array<T, N> const &arr, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    using value_t = std::remove_cvref_t<decltype(reduce(std::declval<T>()))>;
    std::array<value_t, N> res{};
    reduce_range(arr, res, c, root, all, op);
    return res;
  }

  /**
   * @brief Implementation of an MPI reduce for a `std::array` that reduces directly into an existing output array.
   *
   * @details It calls mpi::reduce_range with the input and output array. The output array must be the same size as the
   * input array on receiving ranks.
   *
   * @tparam T1 Value type of the array to be reduced.
   * @tparam N1 Size of the array to be reduced.
   * @tparam T2 Value type of the array to be reduced into.
   * @tparam N2 Size of the array to be reduced into.
   * @param arr_in `std::array` to reduce.
   * @param arr_out `std::array` to reduce into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T1, std::size_t N1, typename T2, std::size_t N2>
  void mpi_reduce_into(std::array<T1, N1> const &arr_in, std::array<T2, N2> &arr_out, communicator c = {}, int root = 0, bool all = false,
                       MPI_Op op = MPI_SUM) {
    reduce_range(arr_in, arr_out, c, root, all, op);
  }

  /** @} */

} // namespace mpi
