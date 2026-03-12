// https://rosettacode.org/wiki/Remove_duplicate_elements#PascalABC.NET

begin
  var a := Arr(1,2,2,2,4,4,6,3,2,2,3,2,7,8,7);
  a.Println;
  a := HSet(a).ToArray;
  a.Println;
end.