##
function Fib: sequence of BigInteger;
begin
  var (a,b) := (1bi,1bi);
  while True do
  begin
    yield a;
    (a,b) := (b,a+b);
  end;
end;

Fib.Take(100).Print