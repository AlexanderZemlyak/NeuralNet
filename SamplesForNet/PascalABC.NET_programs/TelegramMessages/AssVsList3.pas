uses Utils;

var n := 10000000;
var a := ArrRandomInteger(n);

procedure ArrTest;
begin
  var s := 0;
  for var i:=0 to a.Length-1 do
    s += a[i];
end;

begin
  var sw := new Stopwatch;
  sw.Start;
  loop 100 do
    ArrTest;
  sw.Stop;
  Print(sw.ElapsedMilliseconds/100);
end.