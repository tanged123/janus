
#include <iostream>
#include <janus/janus.hpp>

void vector_example() {
    std::cout << "--- Vector Example ---\n";
    // Symbolic Vector
    janus::SymbolicVector v(3);
    auto y = janus::sym("y");
    // Setting elements
    v(0) = y;
    v(1) = y * y;
    v(2) = 1.0;

    std::cout << "Vector v:\n" << v << "\n";

    // Getting elements
    auto v0 = v(0);
    std::cout << "v(0) = " << v0 << "\n";
}

void set_get_example() {
    std::cout << "--- Set/Get Example ---\n";
    janus::SymbolicMatrix M(2, 2);

    // Setting via operator()
    M(0, 0) = 10.0;
    M(0, 1) = janus::sym("a");
    M(1, 0) = janus::sym("b");
    M(1, 1) = M(0, 1) + M(1, 0);

    std::cout << "Matrix M:\n" << M << "\n";

    // Getting via operator()
    auto val = M(1, 1);
    std::cout << "M(1, 1) extracted: " << val << "\n";

    // Block operations (Eigen style)
    M.row(0) = M.row(1);
    std::cout << "After row copy:\n" << M << "\n\n";
}

int main() {
    // Numeric
    janus::JanusMatrix<double> M_num(2, 2);
    M_num << 1.0, 2.0, 3.0, 4.0;
    std::cout << "Numeric Matrix:\n" << M_num << "\n\n";

    // Symbolic
    auto x = janus::sym("x");
    janus::JanusMatrix<janus::SymbolicScalar> M_sym(2, 2);
    M_sym << x, x + 1, x * 2, janus::sin(x);

    std::cout << "Symbolic Matrix (Expression):\n" << M_sym << "\n\n";

    // Symbolic Constant
    janus::JanusMatrix<janus::SymbolicScalar> M_const(2, 2);
    M_const << 1.0, 2.0, 3.0, 4.0;
    std::cout << "Symbolic Matrix (Constant):\n" << M_const << "\n\n";

    set_get_example();
    vector_example();

    return 0;
}
