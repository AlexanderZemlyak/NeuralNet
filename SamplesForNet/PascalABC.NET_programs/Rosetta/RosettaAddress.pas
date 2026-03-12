// https://rosettacode.org/wiki/Address_of_a_variable

begin
  var i: integer := 3;
  var p: ^integer := @i;
  p^ += 2;
  Print(p^,i);
end.