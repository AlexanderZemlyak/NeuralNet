begin
  var a := MatrRandom;
  a.Println;
  // Максимум сумм строк
  a.Rows.Max(row -> row.Sum).Println;
  // Среднее минимумов столбцов
  a.Cols.Average(col -> col.Min).Println;
end.