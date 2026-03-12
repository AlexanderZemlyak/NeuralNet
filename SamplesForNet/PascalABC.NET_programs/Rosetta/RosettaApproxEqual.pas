// https://rosettacode.org/wiki/Approximate_equality#PascalABC.NET

function ApproxEqual(x,y,eps: real): boolean := Abs(x - y) < eps;

begin
  var zoo: Dictionary<string,integer>;
  zoo := Dict(('crocodile',2),('kakadu',3),('elephant',1));
  zoo.Println
end.