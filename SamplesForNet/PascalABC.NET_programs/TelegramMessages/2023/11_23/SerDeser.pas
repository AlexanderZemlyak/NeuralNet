type Arr3 = array of array of array of integer;

begin
  var a := |||1, 2, 3|, |4, 5, 6||, ||7, 8, 9|, |10, 11, 12|||;
  Serialize('a.dat', a);
  a := Arr3(Deserialize('a.dat'));
  Print(a);
end.
