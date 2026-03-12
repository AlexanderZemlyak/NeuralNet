uses Utils;

var n := 100000000;

procedure p(a: array of integer);
begin
  for var i:=0 to a.High do
  begin
    a[i] := 1;
  end;
end;

procedure q(a: array of integer);
begin
  for var i:=0 to a.High step 2 do
  begin
    a[i] := 1;
    a[i+1] := 1;
  end;
end;

begin
  var a := new integer[n];
  Benchmark(()->p(a),10).Println;
  Benchmark(()->q(a),10).Println;
end.