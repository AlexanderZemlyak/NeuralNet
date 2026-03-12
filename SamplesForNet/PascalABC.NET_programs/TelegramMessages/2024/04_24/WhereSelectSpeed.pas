##
var n := 100000000;
var a := ArrRandom(n,0,integer.MaxValue-1);
var b := Copy(a);
Milliseconds;
a := a.Where(x -> x mod 2 = 0).Select(x -> x div 2).ToArray;
Println(MillisecondsDelta);

a := Copy(b);
MillisecondsDelta;
var j := 0;
for var i:=0 to a.Length-1 do
  if a[i] mod 2 = 0 then
  begin  
    a[j] := a[i];
    j += 1;
  end;
SetLength(a,j);

for var i:=0 to a.Length-1 do
  a[i] := a[i] div 2;
Println(MillisecondsDelta);
