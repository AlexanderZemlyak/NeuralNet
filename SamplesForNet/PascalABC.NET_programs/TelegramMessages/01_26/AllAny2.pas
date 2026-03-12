begin
  var a := MatrRandom(5,3);
  a.Println;
  a.Rows.All(row -> row.Any(x -> x.IsEven)).Print;
end.