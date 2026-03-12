uses Utils;

procedure BubbleSortTuple(a: array of integer);
begin
  for var i := 1 to a.Length-1 do
    for var j := a.Length-1 downto i do
      if a[j] < a[j-1] then
        (a[j], a[j-1]) := (a[j-1], a[j])
end;

procedure BubbleSortSwap(a: array of integer);
begin
  for var i := 1 to a.Length-1 do
    for var j := a.Length-1 downto i do
      if a[j] < a[j-1] then
        Swap(a[j], a[j-1]);
end;

begin
  var n := 50000;
  var a := ArrRandomInteger(n,0,integer.MaxValue-1);
  var b := Copy(a);
  Benchmark(procedure -> BubbleSortSwap(b),2).Println;
  b := Copy(a);
  Benchmark(procedure -> BubbleSortTuple(b),2).Println;
end.