{ Тема. Операторы цикла while и repeat

  Задание. Вводятся числа пока не введено число 777
    Сколько чисел до 777 введено?
}
begin
  var count := 0;
  var x := ReadInteger; // первое x вводим до цикла
  while x <> 777 do
  begin
    count += 1;
    x := ReadInteger; // а все следующие - в конце цикла
  end;
  Print(count);
end.
 
