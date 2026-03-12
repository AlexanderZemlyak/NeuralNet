{$reference Mathnet.Numerics.dll}
uses Mathnet.Numerics.LinearAlgebra;

begin
  var A := Matrix&<double>.Build.DenseOfArray(Matr(
    [3.0, 2, -1],
    [2.0, -2, 4],
    [-1, 0.5, -1]));
  var b := Vector&<double>.Build.Dense([1.0, -2, 0]);
  var x := A.Solve(b);
  Print(x.ToArray);
end.