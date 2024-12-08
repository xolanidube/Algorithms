import math
import cmath

class FractalEntanglementFactorization:
    """
    Fractal Entanglement Factorization Algorithm (FEFA)
    Proof-of-concept: Works only for small integers. The polynomial-time claim is theoretical and not proven.
    """

    def __init__(self, branch_factor=2):
        # branch_factor controls the complexity of the fractal base
        self.branch_factor = branch_factor

    def fractal_transform(self, N):
        """
        Create a fractal embedding of N.
        For simplicity, represent N in 'branch_factor' base and map digits to complex points.
        """
        digits = []
        x = N
        while x > 0:
            digits.append(x % self.branch_factor)
            x //= self.branch_factor
        if not digits:
            digits = [0]

        # Map each digit d to a complex point on the unit circle with a fractal offset
        points = []
        for i, d in enumerate(digits):
            angle = 2 * math.pi * d / self.branch_factor + (i * math.pi / (len(digits)+1))
            point = cmath.exp(1j * angle)
            points.append(point)

        return points

    def entanglement_measure(self, points):
        """
        Compute a simplistic entanglement measure: sum of squared distances from mean.
        """
        if not points:
            return 0.0
        mean_x = sum(p.real for p in points) / len(points)
        mean_y = sum(p.imag for p in points) / len(points)
        return sum((p.real - mean_x)**2 + (p.imag - mean_y)**2 for p in points)

    def attempt_factor(self, N):
        """
        Attempt to find a non-trivial factor by 'perturbing' N.
        Hypothetical process: we vary a parameter and look for stable entanglement changes.
        For simplicity, we just try divisors up to sqrt(N) here as a stand-in.

        In the theoretical version, we would do something far more complex:
        - Adjust fractal parameters,
        - Analyze entanglement for patterns,
        - Deduce prime factor.

        Since a true polynomial-time factorization is unproven, we fallback to a classical approach
        to at least show correctness on small numbers.
        """
        limit = int(math.sqrt(N)) + 1
        for f in range(2, limit):
            if N % f == 0:
                return f, N // f
        return None

    def factorize(self, N):
        # Base cases
        if N <= 1:
            return []
        # Check if prime (simple heuristic)
        if self.attempt_factor(N) is None:
            return [N]
        # Otherwise, recurse
        factors = []
        stack = [N]
        while stack:
            curr = stack.pop()
            res = self.attempt_factor(curr)
            if res is None:
                # It's prime
                factors.append(curr)
            else:
                f1, f2 = res
                stack.append(f1)
                stack.append(f2)
        factors.sort()
        return factors


if __name__ == "__main__":
    # Demonstration
    # We show a small test factorization. For N=15, FEFA should find 3 and 5.
    # This does not prove polynomial-time for large N, just correctness for small cases.
    algo = FractalEntanglementFactorization(branch_factor=2)
    test_numbers = [15, 21, 91, 1, 2, 97]
    for num in test_numbers:
        points = algo.fractal_transform(num)
        entanglement = algo.entanglement_measure(points)
        factors = algo.factorize(num)
        print(f"Number: {num}, Factors: {factors}, Entanglement: {entanglement}")
