/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2016 by I. Krivenko
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
#include <mpi/mpi.hpp>
#include <gtest/gtest.h>

using namespace mpi;

TEST(Comm, split) {

  communicator world;
  int rank = world.rank();

  // Skip test if only one rank in communicator
  if (world.size() == 1) return;

  ASSERT_TRUE(2 == world.size() or 4 == world.size());

  int colors[] = {0, 2, 1, 1};
  int keys[]   = {5, 7, 13, 18};

  auto comm = world.split(colors[rank], keys[rank]);

  int comm_sizes[] = {1, 1, 2, 2};
  int comm_ranks[] = {0, 0, 0, 1};

  EXPECT_EQ(comm_sizes[rank], comm.size());
  EXPECT_EQ(comm_ranks[rank], comm.rank());
}

MPI_TEST_MAIN;
