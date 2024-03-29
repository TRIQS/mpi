// Copyright (c) 2019-2022 Simons Foundation
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
// Authors: Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a struct and tags to represent lazy MPI communication.
 */

#pragma once

#include "./communicator.hpp"

#include <mpi.h>

namespace mpi {

  namespace tag {

    /// Tag to specify a lazy MPI reduce call.
    struct reduce {};

    /// Tag to specify a lazy MPI scatter call.
    struct scatter {};

    /// Tag to specify a lazy MPI gather call.
    struct gather {};

  } // namespace tag

  /**
   * @brief Represents a lazy MPI communication.
   *
   * @tparam Tag Type to specify the kind of MPI communication.
   * @tparam T Type to be communicated.
   */
  template <typename Tag, typename T> struct lazy {
    /// Object to be communicated.
    T rhs;

    /// mpi::communicator used in the lazy communication.
    communicator c;

    /// Rank of the root process.
    int root{};

    /// Should we use `MPI_Allxxx`?
    bool all{};

    /// `MPI_Op` used in the lazy communication (only relevant if tag::reduce is used).
    MPI_Op op{};
  };

  // should this be in detail?
  namespace detail {

    /**
     * @brief Type trait to check if a type is mpi::lazy.
     * @tparam T Type to be checked.
     */
    template <typename T> inline constexpr bool is_mpi_lazy = false;

    /**
     * @brief Spezialization of is_mpi_lazy.
     *
     * @tparam Tag Type to specify the kind of MPI call.
     * @tparam T Type to be checked.
     */
    template <typename Tag, typename T> inline constexpr bool is_mpi_lazy<lazy<Tag, T>> = true;

  } // namespace detail

} // namespace mpi
