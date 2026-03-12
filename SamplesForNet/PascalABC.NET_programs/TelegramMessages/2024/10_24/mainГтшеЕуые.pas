uses NUnitABC;
uses MyFuncs;

const eps = 1e-10;

[Test, Combinatorial]
procedure Test4([Values(1,3,5,7)] a: real; 
  [RangeAttribute(1,200)] b: real);
begin
  Assert.AreEqual(Hypot(a, b), Sqrt(a * a + b * b),eps);
end;

begin
end.