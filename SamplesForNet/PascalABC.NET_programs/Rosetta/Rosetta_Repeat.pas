// https://rosettacode.org/wiki/Repeat#PascalABC.NET

procedure Rep(n: integer; p: procedure) := (p * n);

procedure p := Print('Hello');

begin
  Rep(3,p);
end.