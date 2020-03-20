/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2013 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include <mpi/vector.hpp>
#include <gtest/gtest.h>

#include <complex>

using namespace itertools;

TEST(MPI, vector_reduce) {

  mpi::communicator world;

  const int N = 7;
  using VEC   = std::vector<std::complex<double>>;

  VEC A(N), B;

  for (int i = 0; i < N; ++i) A[i] = i; //+ world.rank();

  B = mpi::all_reduce(A, world);

  VEC res(N);
  for (int i = 0; i < N; ++i) res[i] = world.size() * i; // +  world.size()*(world.size() - 1)/2;

  EXPECT_EQ(B, res);
}

// -----------------------------------

TEST(MPI, vector_gather_scatter) {

  mpi::communicator world;

  std::vector<std::complex<double>> A(7), B(7), AA(7);

  for(auto [i, v_i]: enumerate(A)) v_i = i + 1; 

  B = mpi::scatter(A, world);
  auto C = mpi::scatter(A, world);

  for (auto &x : B) x *= -1;
  for (auto &x : AA) x = 0;
  for (auto &x : A) x *= -1;

  AA = mpi::all_gather(B, world);

  EXPECT_EQ(A, AA);
}

MPI_TEST_MAIN;
