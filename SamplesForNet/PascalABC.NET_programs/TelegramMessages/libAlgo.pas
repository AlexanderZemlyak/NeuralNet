library libAlgo;

function GCD(a, b: int64): int64;
begin
  while b <> 0 do
    (a, b) := (b, a mod b);
  Result := Abs(a)
end;

function Digits(n: int64): List<integer>;
begin
  Result := new List<integer>;
  n := Abs(n);
  while n > 0 do
  begin
    Result.Add(n mod 10);
    n := n div 10
  end;
  Result.Reverse
end;
  
end.