begin
  var n := 100000000;
  var pp := (1..n)
    .Select(x -> (Random, Random))
    .Where(p -> Sqr(p[0]) + Sqr(p[1]) < 1)
    .Count / n * 4;
  Print(pp);
end.