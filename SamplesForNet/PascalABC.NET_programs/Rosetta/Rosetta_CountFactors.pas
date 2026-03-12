// https://rosettacode.org/wiki/Count_in_factors#PascalABC.NET

function Factors(N: integer): List<integer>;
begin
  var lst := new List<integer>;
  if N = 1 then
    lst.Add(N);
  var i := 2;
  while i * i <= N do
  begin
    while N.Divs(i) do
    begin
      lst.Add(i);
      N := N div i;
    end;
    i += 1;
  end;
  if N >= 2 then
    lst.Add(N);
  Result := lst;
end;

begin
  foreach var x in (1..10) + (6351..6359) do
    Println($'{x} = {Factors(x).JoinToString('' x '')}');
end.