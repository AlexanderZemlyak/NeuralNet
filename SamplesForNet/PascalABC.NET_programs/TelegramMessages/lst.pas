uses Utils;

var n := 1000000;
var a := ArrRandom(n);

procedure p1();
begin
  var b := a.Where(x -> x > 50).ToArray;
end;

procedure p2();
begin
  var l := new List<integer>;
  for var i:=0 to a.Length-1 do
    if a[i] > 50 then
      l.Add(a[i]);
end;

procedure p3();
begin
  var l := new List<integer>;
  foreach var x in a do
    if x > 50 then
      l.Add(x);
end;

begin
  Benchmark(p1).Println;
  Benchmark(p2).Println;
  Benchmark(p3).Println;
end.