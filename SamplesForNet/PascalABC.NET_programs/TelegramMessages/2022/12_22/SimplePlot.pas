uses PlotWPF;

function IsPrime(x: integer) := (2..x.Sqrt.Round).All(i -> x.NotDivs(i));

begin
  var n := 1000000;
  var a := new real[n+1];
  var cnt := 0;
  for var i:=2 to n do
  begin
    if IsPrime(i) then
      cnt += 1;
    a[i] := cnt;  
  end;
  Print(a[1000],a[10000],a[100000]);
  LineGraphWPF.Create((0..n).Select(x->real(x)),a);
end.
