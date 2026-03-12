{$reference MathNet.Numerics.dll}
uses MathNet.Numerics;

begin
  var a := Generate.Normal(100, 10.0, 2.0);
  Println(a.Take(10));
  Println(a.Mean,a.Median);
end.