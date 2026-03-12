begin
  var a := Arr(1..5);
  var b := Arr(6..10);
  Println(b + a);
  Println(a * 2);
  Println(a * 2 + b * 2);
  Println((a + b) * 2);
  Println(a + 2, TypeName(a + 2)); // результат - последовательность
  Println(a + |2|, TypeName(a + |2|)); // результат - массив
end.