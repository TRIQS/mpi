// Copyright (c) 2022-2024 Simons Foundation
// Copyright (c) 2022 Hugo U.R. Strand
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
// Authors: Thomas Hahn, Hugo U.R. Strand

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <complex>

TEST(MPI, ComplexBroadcast) {
  // broadcast a complex number
  mpi::communicator world;

  std::complex<double> cplx;
  if (world.rank() == 0) cplx = std::complex<double>(1., 2.);

  mpi::broadcast(cplx);

  EXPECT_EQ(cplx, std::complex<double>(1., 2.));
}

MPI_TEST_MAIN;
