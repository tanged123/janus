#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Linalg.hpp> // for to_mx

TEST(JanusTypesTests, SymbolicArgScalar) {
    janus::SymbolicScalar s = 5.0;
    janus::SymbolicArg arg(s);

    EXPECT_FALSE(arg.get().is_empty());
    janus::SymbolicScalar casted = arg; // Implicit conversion
    EXPECT_FALSE(casted.is_empty());
}

TEST(JanusTypesTests, SymbolicArgMatrix) {
    // From Eigen<MX>
    janus::SymbolicMatrix m(2, 2);
    m << 1, 2, 3, 4; // Promotes to MX

    janus::SymbolicArg arg(m);
    casadi::MX mx = arg;

    EXPECT_EQ(mx.size1(), 2);
    EXPECT_EQ(mx.size2(), 2);
}

TEST(JanusTypesTests, SymbolicArgEmpty) {
    janus::SymbolicMatrix empty(0, 0);
    janus::SymbolicArg arg(empty);
    casadi::MX mx = arg;
    EXPECT_TRUE(mx.is_empty());
}

TEST(JanusTypesTests, SymHelpers) {
    auto s = janus::sym("s");
    EXPECT_TRUE(s.is_symbolic());
    EXPECT_EQ(s.name(), "s");

    auto m = janus::sym("m", 2, 3);
    EXPECT_TRUE(m.is_symbolic());
    EXPECT_EQ(m.size1(), 2);
    EXPECT_EQ(m.size2(), 3);
}

TEST(JanusTypesTests, SymVecHelpers) {
    // sym_vec
    auto v = janus::sym_vec("v", 3);
    EXPECT_EQ(v.size(), 3);
    // v is Eigen vector of MX. Check elements are symbolic.
    EXPECT_FALSE(v(0).is_constant());

    // sym_vec_pair
    auto [v_eig, v_mx] = janus::sym_vec_pair("vp", 4);
    EXPECT_EQ(v_eig.size(), 4);
    EXPECT_EQ(v_mx.size1(), 4);
    EXPECT_EQ(v_mx.size2(), 1);

    // as_mx / to_mx consistency check indirectly
    casadi::MX packed = janus::as_mx(v);
    EXPECT_EQ(packed.size1(), 3);
    EXPECT_EQ(packed.size2(), 1);
}

TEST(JanusTypesTests, FixedSizeTypes) {
    janus::Vec3<double> v3;
    v3 << 1, 2, 3;
    EXPECT_EQ(v3.size(), 3);

    janus::Mat2<janus::SymbolicScalar> m2;
    m2 << 1, 2, 3, 4;
    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 2);
}
