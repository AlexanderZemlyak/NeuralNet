begin
  var a := Arr(1..9);
  a.Println;
  var k := 3;
  // Сдвиг влево на k
  a := a[k:]+a[:k];
  a.Println;
end.