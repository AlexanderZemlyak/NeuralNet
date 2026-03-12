begin 
  var x := ReadString;
  var a := Arr('Козлов','Серов','Иванова','Петров','Умнова','Хулиганкин','Белова','Ослов'); 
  a.Println;    
  // Способ 1
  var Есть := False;
  for var i:=0 to a.Length-1 do
    if a[i] = x then
    begin
      Есть := True;
      break;
    end;
  Println(Есть);  
  // Способ 2
  Есть := False;
  foreach var elem in a do
    if elem = x then
    begin
      Есть := True;
      break;
    end;
  Println(Есть);
  // Способ 3
  Println(a.Contains(x));
  // Способ 4
  Println(x in a);
end.
