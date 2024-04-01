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
 * @brief Provides utilities to map C++ datatypes to MPI datatypes.
 */

#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mpi {

  /**
   * @addtogroup mpi_types_ops
   * @{
   */

  namespace detail {

    // Helper function to get the memory displacements of the different elements in a tuple w.r.t. the first element.
    template <typename... T, size_t... Is> void _init_mpi_tuple_displ(std::index_sequence<Is...>, std::tuple<T...> tup, MPI_Aint *disp) {
      ((void)(disp[Is] = {(char *)&std::get<Is>(tup) - (char *)&std::get<0>(tup)}), ...);
    }

  } // namespace detail

  /**
   * @brief Map C++ datatypes to the corresponding MPI datatypes.
   *
   * @details C++ types which have a corresponding MPI datatype should specialize this struct. It is assumed that it
   * has a static member function `get` which returns the `MPI_Datatype` object for a given C++ type. For example:
   *
   * @code{.cpp}
   * template <> struct mpi_type<int> {
   *   static MPI_Datatype get() noexcept { return MPI_INT; }
   * }
   * @endcode
   *
   * @tparam T C++ datatype.
   */
  template <typename T> struct mpi_type {};

#define D(T, MPI_TY)                                                                                                                                 \
  /** @brief Specialization of mpi_type for T. */                                                                                                    \
  template <> struct mpi_type<T> {                                                                                                                   \
    [[nodiscard]] static MPI_Datatype get() noexcept { return MPI_TY; }                                                                              \
  }

  // mpi_type specialization for various built-in types
  D(bool, MPI_C_BOOL);
  D(char, MPI_CHAR);
  D(int, MPI_INT);
  D(long, MPI_LONG);
  D(long long, MPI_LONG_LONG);
  D(double, MPI_DOUBLE);
  D(float, MPI_FLOAT);
  D(std::complex<double>, MPI_C_DOUBLE_COMPLEX);
  D(unsigned long, MPI_UNSIGNED_LONG);
  D(unsigned int, MPI_UNSIGNED);
  D(unsigned long long, MPI_UNSIGNED_LONG_LONG);
#undef D

  /**
   * @brief Specialization of mpi::mpi_type for `const` types.
   * @tparam T C++ type.
   */
  template <typename T> struct mpi_type<const T> : mpi_type<T> {};

  /**
   * @brief Type trait to check if a type T has a corresponding MPI datatype, i.e. if mpi::mpi_type has been specialized.
   * @tparam T Type to be checked.
   */
  template <typename T, typename = void> constexpr bool has_mpi_type = false;

  /**
   * @brief Specialization of mpi::has_mpi_type for types which have a corresponding MPI datatype.
   * @tparam T Type to be checked.
   */
  template <typename T> constexpr bool has_mpi_type<T, std::void_t<decltype(mpi_type<T>::get())>> = true;

  /**
   * @brief Create a new `MPI_Datatype` from a tuple.
   *
   * @details The tuple element types must have corresponding MPI datatypes, i.e. they must have mpi::mpi_type
   * specializtions. It uses `MPI_Type_create_struct` to create a new datatype consisting of the tuple element types.
   *
   * @tparam Ts Tuple element types.
   * @param tup Tuple object.
   * @return `MPI_Datatype` consisting of the types of the tuple elements.
   */
  template <typename... Ts> [[nodiscard]] MPI_Datatype get_mpi_type(std::tuple<Ts...> tup) {
    static constexpr int N            = sizeof...(Ts);
    std::array<MPI_Datatype, N> types = {mpi_type<std::remove_reference_t<Ts>>::get()...};

    // the number of elements per type (we want 1 per type)
    std::array<int, N> blocklen;
    for (int i = 0; i < N; ++i) { blocklen[i] = 1; }

    // displacements of the blocks in bytes w.r.t. to the memory address of the first block
    std::array<MPI_Aint, N> disp;
    detail::_init_mpi_tuple_displ(std::index_sequence_for<Ts...>{}, tup, disp.data());
    if (std::any_of(disp.begin(), disp.end(), [](MPI_Aint i) { return i < 0; })) {
      std::cerr << "ERROR: Custom mpi types require non-negative displacements\n";
      std::abort();
    }

    // create and return MPI datatype
    MPI_Datatype cty{};
    MPI_Type_create_struct(N, blocklen.data(), disp.data(), types.data(), &cty);
    MPI_Type_commit(&cty);
    return cty;
  }

  /**
   * @brief Specialization of mpi::mpi_type for std::tuple.
   * @tparam Ts Tuple element types.
   */
  template <typename... T> struct mpi_type<std::tuple<T...>> {
    [[nodiscard]] static MPI_Datatype get() noexcept { return get_mpi_type(std::tuple<T...>{}); }
  };

  /**
   * @brief Create an `MPI_Datatype` from some struct.
   *
   * @details It is assumed that there is a free function `tie_data` which returns a tuple containing the data
   * members of the given type. The intended use is as a base class for a specialization of mpi::mpi_type:
   *
   * @code{.cpp}
   * // type to use for MPI communication
   * struct foo {
   *   double x;
   *   int y;
   * };
   *
   * // provide a tie_data function
   * auto tie_data(foo f) {
   *   return std::tie(f.x, f.y);
   * }
   *
   * // provide a specialization of mpi_type
   * template <> struct mpi::mpi_type<foo> : mpi::mpi_type_from_tie<foo> {};
   * @endcode
   *
   * @tparam T Type to be converted to an `MPI_Datatype`.
   */
  template <typename T> struct mpi_type_from_tie {
    [[nodiscard]] static MPI_Datatype get() noexcept { return get_mpi_type(tie_data(T{})); }
  };

  /** @} */

} // namespace mpi
