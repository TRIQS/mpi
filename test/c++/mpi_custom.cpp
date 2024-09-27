// Copyright (c) 2020-2024 Simons Foundation
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
// Authors: Thomas Hahn, Nils Wentzell

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>
#include <mpi.h>

#include <array>
#include <tuple>

// a simple struct representing a complex number
struct custom_cplx {
  double real{}, imag{};

  // add two custom_cplx objects
  custom_cplx operator+(custom_cplx z) const {
    z.real += real;
    z.imag += imag;
    return z;
  }

  // default equal-to operator
  bool operator==(const custom_cplx &) const = default;
};

// tie the data (used to construct the custom MPI type)
inline auto tie_data(custom_cplx z) { return std::tie(z.real, z.imag); }

// specialize mpi_type for custom_cplx
template <> struct mpi::mpi_type<custom_cplx> : mpi::mpi_type_from_tie<custom_cplx> {};

// stand-alone add function (the same as the operator+ above)
custom_cplx add(custom_cplx const &x, custom_cplx const &y) { return x + y; }

TEST(MPI, CustomTypeMapAdd) {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int root = 0;

  // array of custom_cplx objects to be reduced
  std::array<custom_cplx, 10> arr;
  for (int i = 0; i < 10; ++i) {
    arr[i].real = rank + 1;
    arr[i].imag = 0;
  }

  // reduce and check result
  auto reduced_arr = mpi::reduce(arr, world, root, false, mpi::map_add<custom_cplx>());
  if (rank == root)
    for (auto &z : reduced_arr) { ASSERT_NEAR(z.real, size * (size + 1) / 2.0, 1.e-14); }
}

TEST(MPI, CustomTypeMapCFunction) {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int root = 0;

  // array of custom_cplx objects to be reduced
  std::array<custom_cplx, 10> arr;
  for (int i = 0; i < 10; ++i) {
    arr[i].real = rank + 1;
    arr[i].imag = 0;
  }

  // reduce and check result
  auto reduced_arr = mpi::reduce(arr, world, root, false, mpi::map_C_function<custom_cplx, add>());
  if (rank == root)
    for (auto &z : reduced_arr) { ASSERT_NEAR(z.real, size * (size + 1) / 2.0, 1.e-14); }
}

// custom tuple type
using tuple_type = std::tuple<int, long long, double>;

// two tuples are summed by adding the first and last component, the middle one is zeroed
auto add(tuple_type const &lhs, tuple_type const &rhs) {
  return std::make_tuple(std::get<0>(lhs) + std::get<0>(rhs), 0ll, std::get<2>(lhs) + std::get<2>(rhs));
}

TEST(MPI, TupleTypeMapCFunction) {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int root = 0;

  // array of custom_type objects to be reduced
  std::array<tuple_type, 10> arr;
  for (auto &tup : arr) {
    std::get<0>(tup) = rank + 1;
    std::get<2>(tup) = 3.0 * (rank + 1);
  }

  // reduce and check result
  auto reduced_arr = mpi::reduce(arr, world, root, false, mpi::map_C_function<tuple_type, add>());
  // MPI_Reduce(arr.data(), reduced_arr.data(), 10, mpi_type<tuple_type>::get(), mpi::map_C_function<tuple_type, add>(), root, MPI_COMM_WORLD);
  if (rank == root)
    for (int i = 0; i < 10; ++i) { ASSERT_NEAR(std::get<0>(reduced_arr[i]) + std::get<2>(reduced_arr[i]), 2 * size * (size + 1), 1.e-14); }
}

TEST(MPI, TupleMPIDatatypes) {
  mpi::communicator world;
  int rank = world.rank();
  int root = 0;

  // check custom MPI datatypes of various tuple types
  using type1 = std::tuple<int>;
  type1 tup1;
  if (rank == root) { tup1 = std::make_tuple(100); }
  mpi::broadcast(tup1, world, root);
  EXPECT_EQ(tup1, std::make_tuple(100));

  using type2 = std::tuple<int, double>;
  type2 tup2;
  if (rank == root) { tup2 = std::make_tuple(100, 3.1314); }
  mpi::broadcast(tup2, world, root);
  EXPECT_EQ(tup2, std::make_tuple(100, 3.1314));

  using type5 = std::tuple<int, double, char, custom_cplx, bool>;
  type5 tup5;
  if (rank == root) { tup5 = std::make_tuple(100, 3.1314, 'r', custom_cplx{1.0, 2.0}, false); }
  mpi::broadcast(tup5, world, root);
  EXPECT_EQ(tup5, std::make_tuple(100, 3.1314, 'r', custom_cplx{1.0, 2.0}, false));
}

MPI_TEST_MAIN;
