function Fib(n: integer; a: integer := 1; b: integer := 1): integer;
begin
  if n = 1 then
    Result := a
  else Result := Fib(n-1,b,a+b)
end;

begin
  Fib(30).Print;
end.