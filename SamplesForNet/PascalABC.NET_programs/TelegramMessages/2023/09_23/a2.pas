begin
  // Описание, выделение памяти и заполнение 
  // Компиляятор автоматически выводит тип массива
  var a := Arr(5,3,2,5,4);
  
  // Вывод - цикл foreach
  foreach var x in a do
    Print(x);
  
  // Вывод - метод Print
  a.Println
end.