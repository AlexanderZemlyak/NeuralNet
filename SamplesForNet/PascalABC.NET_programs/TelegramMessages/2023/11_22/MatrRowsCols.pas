begin
  var a := MatrRandomInteger(6,9,2,5);
  a.Println(3);
  // Количество двоек и троек в 0 строке
  a.Row(0).Count(x -> x in |2,3|).Println;
  // Количество двоек и троек в 0 строке двумерным срезом
  a[0,:].Count(x -> x in |2,3|).Println;
  // Количество двоек и троек в каждой строке (массовый запрос)
  ArrGen(a.RowCount, i -> a.Row(i).Count(x -> x in |2,3|)).Println;
  // Сумма значений каждом столбце (массовый запрос)
  ArrGen(a.ColCount, i -> a.Col(i).Sum).Println;
end.

