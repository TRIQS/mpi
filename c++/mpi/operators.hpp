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
 * @brief Provides utilities to map custom binary functions to MPI operators.
 */

#pragma once

#include <mpi.h>

namespace mpi {

  /**
   * @addtogroup mpi_types_ops
   * @{
   */

  namespace detail {

    // Lambda that maps a binary user function to an `MPI_User_function`.
    template <typename T, T (*F)(T const &, T const &)>
    // unary plus converts the lambda to a function pointer
    MPI_User_function *_map_function = +[](void *in, void *inout, int *len, MPI_Datatype *) { // NOLINT (MPI_Op_create needs a non-const pointer)
      auto *inT    = static_cast<T *>(in);
      auto *inoutT = static_cast<T *>(inout);
      for (int i = 0; i < *len; ++i, ++inT, ++inoutT) { *inoutT = F(*inoutT, *inT); }
    };

    // Generic addition.
    template <typename T> T _generic_add(T const &lhs, T const &rhs) { return lhs + rhs; }

  } // namespace detail

  /**
   * @brief Create a new `MPI_Op` from a given binary function by calling `MPI_Op_create`.
   *
   * @details The binary function must have the following signature `(T const&, T const&) -> T`.
   *
   * @tparam T Type on which the binary function operates.
   * @tparam F Binary function pointer to be mapped.
   * @return `MPI_Op` created from the binary function.
   */
  template <typename T, T (*F)(T const &, T const &)> MPI_Op map_C_function() {
    MPI_Op myOp{};
    MPI_Op_create(detail::_map_function<T, F>, true, &myOp);
    return myOp;
  }

  /**
   * @brief Create a new `MPI_Op` for a generic addition by calling `MPI_Op_create`.
   *
   * @details The type is required to have an overloaded `operator+(const T& lhs, const T& rhs) -> T`.
   *
   * @tparam T Type used for the addition.
   * @return `MPI_Op` for the generic addition of the given type.
   */
  template <typename T> MPI_Op map_add() {
    MPI_Op myOp{};
    MPI_Op_create(detail::_map_function<T, detail::_generic_add<T>>, true, &myOp);
    return myOp;
  }

} // namespace mpi
