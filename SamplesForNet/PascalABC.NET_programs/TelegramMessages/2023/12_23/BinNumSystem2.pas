{ Тема. Операторы цикла while и repeat. Цифры числа

  Задание. Дано целое число x. Сколько единиц в его записи 
    в двоичной системе счисления?
}
begin
  var x := ReadInteger;
  var count := 0;
  while x <> 0 do
  begin
    var d := x mod 2;
    Print(d);
    x := x div 2;
    if d = 1 then
      count += 1
  end;
  Println;
  Print(count);
end.
