// https://rosettacode.org/wiki/First_power_of_2_that_has_leading_decimal_digits_of_12#PascalABC.NET

function p(L,n: integer): integer;
begin
  var logof2 := log10(2);
  var places := trunc(log10(L));
  var nfound := 0;
  var i := 1;
  while True do
  begin
    var a := i * logof2;
    var b := trunc(power(10,a-trunc(a)+places));
    if L = b then
    begin
      nfound += 1;
      if nfound = n then break
    end;
    i += 1;
  end;
  Result := i;
end;

begin
  foreach var (n,L)in Arr() do
    Println($'p({n},{L}) = {p(n, L)}')
end.