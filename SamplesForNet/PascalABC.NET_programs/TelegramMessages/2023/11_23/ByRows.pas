function ByRows(a: array [,] of integer; f: (array of integer) -> integer)
  : array of integer
    := a.Rows.Select(row -> f(row)).ToArray;

begin
  var m := MatrRandom(3,4);
  m.Println;
  ByRows(m,a->a.Sum).Print;
end.
