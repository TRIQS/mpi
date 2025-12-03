// Copyright (c) 2024 Simons Foundation
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
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides utilities to map C++ datatypes to MPI datatypes.
 */

#pragma once

#include "./utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mpi {

  /**
   * @addtogroup mpi_types_ops
   * @{
   */

  /**
   * @brief Map C++ datatypes to the corresponding MPI datatypes.
   *
   * @details C++ types which have a corresponding MPI datatype should specialize this struct. It is assumed that it
   * has a static member function `get` which returns the `MPI_Datatype` object for a given C++ type.
   *
   * For example:
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
  /** @brief Specialization of mpi_type for `T`. */                                                                                                  \
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
   * @brief Specialization of mpi::mpi_type for enum types.
   * @tparam E C++ enum type.
   */
  template <typename E>
    requires(std::is_enum_v<E>)
  struct mpi_type<E> : mpi_type<std::underlying_type_t<E>> {};

  /**
   * @brief Specialization of mpi::mpi_type for `const` types.
   * @tparam T C++ type.
   */
  template <typename T> struct mpi_type<const T> : mpi_type<T> {};

  /**
   * @brief Type trait to check if a type `T` has a corresponding MPI datatype, i.e. if mpi::mpi_type has been
   * specialized.
   *
   * @tparam T Type to be checked.
   */
  template <typename T> constexpr bool has_mpi_type = requires { mpi_type<T>::get(); };

  namespace detail {

    // Helper struct to check if member types are mpi-serializable, i.e. have an associated mpi_type
    struct serialize_checker {
      template <typename T>
      void operator&(T &)
        requires(has_mpi_type<T>)
      {}
    };

  } // namespace detail

  /**
   * @brief A concept that checks if objects of a type can be serialized and deserialized.
   * @tparam T Type to check.
   */
  template <typename T>
  concept Serializable = requires(const T ac, T a, detail::serialize_checker ar) {
    { ac.serialize(ar) } -> std::same_as<void>;
    { a.deserialize(ar) } -> std::same_as<void>;
  };

  /**
   * @brief Create a new `MPI_Datatype` from a tuple.
   *
   * @details The tuple element types must have corresponding MPI datatypes, i.e. they must have mpi::mpi_type
   * specializtions. It uses `MPI_Type_create_struct` to create a new datatype consisting of the tuple element types.
   *
   * The success of MPI calls is checked with mpi::check_mpi_call.
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
    // initialize displacement array from the tuple element addresses
    []<size_t... Is>(std::index_sequence<Is...>, auto &t, MPI_Aint *d) {
      ((d[Is] = (char *)&std::get<Is>(t) - (char *)&std::get<0>(t)), ...);
      // account for non-trivial memory layouts of the tuple elements
      auto min_el = *std::min_element(d, d + sizeof...(Ts));
      ((d[Is] -= min_el), ...);
    }(std::index_sequence_for<Ts...>{}, tup, disp.data());

    // create and return MPI datatype
    MPI_Datatype cty{};
    check_mpi_call(MPI_Type_create_struct(N, blocklen.data(), disp.data(), types.data(), &cty), "MPI_Type_create_struct");
    check_mpi_call(MPI_Type_commit(&cty), "MPI_Type_commit");
    return cty;
  }

  /**
   * @brief Specialization of mpi::mpi_type for `std::tuple`.
   * @tparam Ts Tuple element types.
   */
  template <typename... Ts> struct mpi_type<std::tuple<Ts...>> {
    [[nodiscard]] static MPI_Datatype get() noexcept {
      static MPI_Datatype type = get_mpi_type(std::tuple<Ts...>{});
      return type;
    }
  };

  /**
   * @brief Create an `MPI_Datatype` from some struct.
   *
   * @details It is assumed that there is a free function `tie_data` which returns a tuple containing the data members 
   * of the given type:
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
   * @endcode
   *
   * @tparam U Type to be converted to an `MPI_Datatype`.
   */
  template <typename U>
    requires(not Serializable<U>) and requires(U u) { tie_data(u); }
  struct mpi_type<U> {
    [[nodiscard]] static MPI_Datatype get() noexcept {
      static MPI_Datatype type = get_mpi_type(tie_data(U{}));
      return type;
    }
  };

  namespace detail {

    // Archive helper class to obtain MPI custom type info using references to class members.
    struct mpi_archive {
      std::vector<int> block_lengths{};
      std::vector<MPI_Aint> displacements{};
      std::vector<MPI_Datatype> types{};
      MPI_Aint base_address{};

      // Constructor sets the base address of the object.
      explicit mpi_archive(const void *base) { MPI_Get_address(base, &base_address); }

      // Overloaded operator& to process members to set the block lengths, displacements and MPI types.
      template <typename T>
        requires(has_mpi_type<T>)
      mpi_archive &operator&(const T &member) {
        types.push_back(mpi_type<T>::get());
        MPI_Aint address{};
        MPI_Get_address(&member, &address);
        displacements.push_back(MPI_Aint_diff(address, base_address));
        block_lengths.push_back(1);
        return *this;
      }
    };

  } // namespace detail

  /**
   * @brief Create an `MPI_Datatype` from a serializable type.
   *
   * @details It is assumed that the type has a member function `serialize` which feeds all its class members into an 
   * archive using the `operator&`.
   *
   * @code{.cpp}
   * // type to use for MPI communication
   * struct foo {
   *   double x;
   *   int y;
   *   void serialize(auto& ar) const { ar & x & y; }
   *   void deserialize(auto& ar) { ar & x & y; }
   * };
   * @endcode
   *
   * @tparam T Type to be converted to an `MPI_Datatype`.
   */
  template <Serializable T> [[nodiscard]] MPI_Datatype get_mpi_type(const T &obj) {
    detail::mpi_archive ar(&obj);
    obj.serialize(ar);
    MPI_Datatype mpi_type{};
    MPI_Type_create_struct(static_cast<int>(ar.block_lengths.size()), ar.block_lengths.data(), ar.displacements.data(), ar.types.data(), &mpi_type);
    MPI_Type_commit(&mpi_type);
    return mpi_type;
  }

  /**
   * @brief Specialization of mpi::mpi_type for serializable types.
   * @tparam S Serializable type.
   */
  template <Serializable S> struct mpi_type<S> {
    [[nodiscard]] static MPI_Datatype get() noexcept {
      static MPI_Datatype type = get_mpi_type(S{});
      return type;
    }
  };

  /** @} */

} // namespace mpi
