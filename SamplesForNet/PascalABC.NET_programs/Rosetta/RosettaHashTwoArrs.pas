// https://rosettacode.org/wiki/Hash_from_two_arrays#PascalABC.NET

begin
  var Keys := Arr('aa','bb','cc');
  var Values := Arr(1..3);
  var dct := Dict(Keys.Zip(Values));
  dct.Println;
end.