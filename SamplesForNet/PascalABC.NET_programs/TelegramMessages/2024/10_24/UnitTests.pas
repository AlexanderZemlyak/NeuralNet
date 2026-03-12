uses NUnitABC;

function IsPrime(n: integer): boolean;
begin
  Result := True;
  for var i:=2 to n.Sqrt.Round do
    if n.Divs(i) then
    begin
      Result := False;
      exit
    end;
end;

[TestCase(2)]
[TestCase(3)]
[TestCase(5)]
procedure TestPrime1(n: integer);
begin
  Assert.IsTrue(IsPrime(n));
end;

begin
end.