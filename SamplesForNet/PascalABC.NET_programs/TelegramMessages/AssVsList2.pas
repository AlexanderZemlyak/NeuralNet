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

procedure ArrTest2;
begin
  var s := 0;
  for var i:=0 to n-1 step 4 do  
  begin  
    s += a[i];
    s += a[i+1];
    s += a[i+2];
    s += a[i+3];
  end;  
end;
begin
  Benchmark(ArrTest).Println;
  Benchmark(ArrTest2).Println;
  //Benchmark(()->begin
  //   var s := a.AsParallel.Sum;
  //end).Println;
end.