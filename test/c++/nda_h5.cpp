#define NDA_ENFORCE_BOUNDCHECK
#include "./test_common.hpp"
#include <h5/h5.hpp>

// FIXME  RENAME THIS FILE
#include <nda/h5/simple_read_write.hpp>
// ==============================================================

template <typename T>
void one_test(std::string name, T scalar) {

  nda::array<T, 1> a(5), a_check;
  nda::array<T, 2> b(2, 3), b_check;
  nda::array<T, 3> c(2, 3, 4), c_check;

  for (int i = 0; i < 5; ++i) { a(i) = scalar * (10 * i); }

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) { b(i, j) = scalar * (10 * i + j); }

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) { c(i, j, k) = scalar * (i + 10 * j + 100 * k); }

  std::string filename = "ess_" + name + ".h5";
  // WRITE the file
  {
    h5::file file(filename, 'w');
    h5::group top(file);
    top.create_group("G");

    h5_write(top, "A", a);
    h5_write(top, "B", b);
    h5_write(top, "C", c);
    h5_write(top, "scalar", scalar);

    // add some attribute to A
    auto id = top.open_dataset("A");
    h5_write_attribute(id, "AttrOfA1", 12);
    h5_write_attribute(id, "AttrOfA2", 8.9);

    // in a subgroup
    auto G = top.open_group("G");
    h5_write(G, "A2", a);
  }

  // READ the file
  {
    h5::file file(filename, 'r');
    h5::group top(file);

    h5_read(top, "A", a_check);
    EXPECT_EQ_ARRAY(a, a_check);

    h5_read(top, "B", b_check);
    EXPECT_EQ_ARRAY(b, b_check);

    h5_read(top, "C", c_check);
    EXPECT_EQ_ARRAY(c, c_check);

    EXPECT_EQ(scalar, h5::h5_read<T>(top, "scalar"));

    auto d = a;
    d()    = 0;
    h5_read(top, "G/A2", d);
    EXPECT_EQ_ARRAY(a, d);

    // read the attributes of A
    auto id     = top.open_dataset("A");
    int att1    = h5::h5_read_attribute<int>(id, "AttrOfA1");
    double att2 = h5::h5_read_attribute<double>(id, "AttrOfA2");
    EXPECT_EQ(att1, 12);
    EXPECT_EQ(att2, 8.9);
  }
}

//------------------------------------
TEST(Basic, Int) { one_test<int>("int", 1); }

TEST(Basic, Long) { one_test<long>("long", 1); }

TEST(Basic, Double) { one_test<double>("double", 1.5); }

TEST(Basic, Dcomplex) { one_test<dcomplex>("dcomplex", (1.0 + 1.0i)); }

//------------------------------------

TEST(Basic, Empty) {

  nda::array<long, 2> a(0, 10);
  {
    h5::file file("ess_empty.h5", 'w');
    //h5::group top(file);
    h5_write(file, "A", a);
  }
  {
    h5::file file("ess.h5", 'r');
    h5::group top(file);
    nda::array<double, 2> empty(5, 5);
    h5_read(top, "empty", empty);
    EXPECT_EQ_ARRAY(empty, (nda::array<double, 2>(0, 10)));
  }
}

//------------------------------------

TEST(Basic, String) {

  {
    h5::file file("ess_string.h5", 'w');
    h5_write(file, "s", std::string("a nice chain"));
    h5_write(file, "sempty", "");
  }
  {
    h5::file file("ess_string.h5", 'r');
    nda::array<double, 2> empty(5, 5);

    std::string s2("----------------------------------");
    h5_read(file, "s", s2);
    EXPECT_EQ(s2, "a nice chain");

    std::string s3; //empty
    h5_read(file, "s", s3);
    EXPECT_EQ(s3, "a nice chain");

    std::string s4; //empty
    h5_read(file, "sempty", s4);
    EXPECT_EQ(s4, "");
  }
}

//------------------------------------

TEST(Array, H5) {

  nda::array<long, 2> A(2, 3), B;
  nda::array<double, 2> D(2, 3), D2;
  nda::array<dcomplex, 1> C(5), C2;
  //dcomplex z(1, 2);

  for (int i = 0; i < 5; ++i) C(i) = dcomplex(i, i);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      A(i, j) = 10 * i + j;
      D(i, j) = A(i, j) / 10.0;
    }

  // WRITE the file
  {
    h5::file file("ess.h5", 'w');
    h5::group top(file);

    h5_write(top, "A", A);
    h5_write(top, "C", C);
    h5_write(top, "D", D);
    h5::h5_write(top, "S", "");
    h5_write(top, "A_slice", A(nda::range(), nda::range(1, 3)));
    h5_write(top, "empty", nda::array<double, 2>(0, 10));

    // add some attribute to A
    auto id = top.open_dataset("A");
    h5_write_attribute(id, "AttrOfA1", 12);
    h5_write_attribute(id, "AttrOfA2", 8.9);

    // scalar
    double x = 2.3;
    h5_write(top, "x", x);

    // dcomplex xx(2, 3);
    // h5_write(top, "xx", xx);

    h5_write(top, "s", std::string("a nice chain"));

    top.create_group("G");
    h5_write(top, "G/A", A);

    auto G = top.open_group("G");
    h5_write(G, "A2", A);
  }

  // READ the file
  {
    h5::file file("ess.h5", 'r');
    h5::group top(file);

    h5_read(top, "A", B);
    EXPECT_EQ_ARRAY(A, B);

    // read the attributes of A
    auto id     = top.open_dataset("A");
    int att1    = h5::h5_read_attribute<int>(id, "AttrOfA1");
    double att2 = h5::h5_read_attribute<double>(id, "AttrOfA2");

    EXPECT_EQ(att1, 12);
    EXPECT_EQ(att2, 8.9);

    h5_read(top, "D", D2);
    EXPECT_ARRAY_NEAR(D, D2);

    h5_read(top, "C", C2);
    EXPECT_ARRAY_NEAR(C, C2);

    nda::array<long, 2> a_sli;
    h5_read(top, "A_slice", a_sli);
    EXPECT_EQ_ARRAY(a_sli, A(nda::range(), nda::range(1, 3)));

    double xxx = 0;
    h5_read(top, "x", xxx);
    EXPECT_DOUBLE_EQ(xxx, 2.3);

    std::string s2("----------------------------------");
    h5_read(top, "s", s2);
    EXPECT_EQ(s2, "a nice chain");

    nda::array<double, 2> empty(5, 5);
    h5_read(top, "empty", empty);
    EXPECT_EQ_ARRAY(empty, (nda::array<double, 2>(0, 10)));
  }
}

// ==============================================================

TEST(Vector, String) {

  // vector of string
  std::vector<std::string> V1, V2;
  V1.push_back("abcd");
  V1.push_back("de");

  // writing
  h5::file file("test_nda::array_string.h5", 'w');
  h5::group top(file);

  h5_write(top, "V", V1);

  // rereading
  h5_read(top, "V", V2);

  //comparing
  for (int i = 0; i < 2; ++i) { EXPECT_EQ(V1[i], V2[i]); }
}

/*
  // ==============================================================

TEST(Array, H5ArrayString) {

  // nda::array of string
  nda::array<std::string, 1> A(2), B;
  A(0) = "Nice String";
  A(1) = "another";

  // vector of string
  std::vector<std::string> V1, V2;
  V1.push_back("abcd");
  V1.push_back("de");

  // writing
  h5::file file("test_nda::array_string.h5", 'w');
  h5::group top(file);

  h5_write(top, "A", A);
  h5_write(top, "V", V1);

  // rereading
  h5_read(top, "A", B);
  h5_read(top, "V", V2);

  //comparing
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(V1[i], V2[i]);
    EXPECT_EQ(A(i), B(i));
  }
}
*/
// ==============================================================

// -----------------------------------------------------
// Testing the loading of nda::array of double into complex
// -----------------------------------------------------
/*
TEST(Array, H5RealIntoComplex) {

  nda::array<double, 2> D(2, 3);
  nda::array<dcomplex, 2> C(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) { D(i, j) = 10 * i + j; }

  // WRITE the file
  {
    h5::file file("ess_real_complex.h5", 'w');
    h5::group top(file);
    h5_write(top, "D", D);
  }

  // READ the file
  {
    C() = 89.0 + 9i; // put garbage in it
    h5::file file("ess_real_complex.h5", 'r');
    h5::group top(file);
    h5_read(top, "D", C);
    EXPECT_ARRAY_NEAR(C, D);
  }
}
*/
// ==============================================================

// -----------------------------------------------------
// Testing h5 for std vector
// -----------------------------------------------------

TEST(Array, H5StdVector) {

  std::vector<double> v{1.1, 2.2, 3.3, 4.5};
  std::vector<std::complex<double>> vc{1.1, 2.2, 3.3, 4.5};

  std::vector<double> v2;
  std::vector<std::complex<double>> vc2;

  {
    h5::file file1("test_std_vector.h5", 'w');
    // do we need this top ?
    h5::group top(file1);
    h5_write(top, "vdouble", v);
    h5_write(top, "vcomplex", vc);
  }

  {
    h5::file file2("test_std_vector.h5", 'r');
    h5::group top2(file2);
    h5_read(top2, "vdouble", v2);
    h5_read(top2, "vcomplex", vc2);
  }

  for (size_t i = 0; i < v.size(); ++i) EXPECT_EQ(v[i], v2[i]);
  for (size_t i = 0; i < vc.size(); ++i) EXPECT_EQ(vc[i], vc2[i]);
}

// ==============================================================

// -----------------------------------------------------
// Testing h5 for an nda::array of matrix
// -----------------------------------------------------

TEST(BlockMatrixH5, S1) {

  using mat_t = nda::array<double, 2>;
  nda::array<mat_t, 1> W, V{mat_t{{1, 2}, {3, 4}}, mat_t{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}};

  {
    h5::file file1("ess_non_pod.h5", 'w');
    h5_write(file1, "block_mat", V);
  }

  {
    h5::file file2("ess_non_pod.h5", 'r');
    h5_read(file2, "block_mat", W);
  }

  EXPECT_EQ(first_dim(V), first_dim(W));
  for (int i = 0; i < first_dim(V); ++i) EXPECT_ARRAY_NEAR(V(i), W(i));
}
