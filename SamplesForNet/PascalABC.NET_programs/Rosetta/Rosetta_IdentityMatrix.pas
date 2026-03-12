// https://rosettacode.org/wiki/Identity_matrix#PascalABC.NET

begin
  var n := ReadInteger;
  var matrix := MatrGen(n,n,(i,j) -> i = j ? 1 : 0);
  matrix.Println
end.