// https://rosettacode.org/wiki/Find_limit_of_recursion#PascalABC.NET

procedure Recur(i: integer);
begin
  System.Console.WriteLine(i);
  Recur(i + 1);
end;

begin
  Recur(0);
end.