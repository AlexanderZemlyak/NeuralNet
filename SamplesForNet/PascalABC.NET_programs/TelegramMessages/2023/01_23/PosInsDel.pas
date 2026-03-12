// Заменить второе вхождение подстроки 'оранжевый' на 'малиновый'
begin
  var s := 'красный оранжевый синий оранжевый черный оранжевый';
  Println(s);
  var s1 := 'оранжевый';
  var s2 := 'малиновый';
  var p := Pos(s1,s);
  p := Pos(s1,s,p+1);
  Delete(s,p,s1.Length);
  Insert(s2,s,p);
  Println(s);
end.
