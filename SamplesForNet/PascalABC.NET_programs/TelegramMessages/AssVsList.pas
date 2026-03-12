uses Utils;

var n := 1000000;
var a := ArrRandomInteger(n);
var L := a.ToList;

procedure ArrTest;
begin
  var s := 0;
  for var i:=0 to a.Length-1 do
    s += a[i];
end;

procedure ArrTestForeach;
begin
  var s := 0;
  foreach var x in a do
    s += x;
end;

procedure LstTest;
begin
  var s := 0;
  for var i:=0 to L.Count-1 do
    s += L[i];
end;

procedure LstTestForeach;
begin
  var s := 0;
  foreach var x in L do
    s += x;
end;

var tmp: integer := 666;
procedure ArrSumTest := tmp := a.Sum;
procedure LstSumTest := tmp := L.Sum;

begin
  Benchmark(ArrTest).Println;
  Benchmark(ArrTestForeach).Println;
//  Benchmark(LstTest).Println;
  Benchmark(ArrSumTest).Println;
//  Benchmark(LstSumTest).Println;
end.