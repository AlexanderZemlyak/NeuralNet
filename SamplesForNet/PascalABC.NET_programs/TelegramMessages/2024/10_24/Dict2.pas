begin
  var grades := Dict(
    'Иван' to Arr(5, 4, 3),
    'Анна' to Arr(5, 5, 4),
    'Пётр' to Arr(3, 2, 4)
  );  

  var averageGrades := new Dictionary<string, real>;

  foreach var pair in grades do
    averageGrades[pair.Key] := pair.Value.Average;

  averageGrades.PrintLines(pair -> $'{pair.Key} -> {pair.Value,0:f2}');
end.