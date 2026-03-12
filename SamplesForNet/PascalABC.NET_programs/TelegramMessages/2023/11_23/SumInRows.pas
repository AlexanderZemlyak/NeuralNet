begin
  var a := MatrRandomInteger(3,4);  
  a.Println;
  a.Rows.ConvertAll(row -> row.Sum).Println;
end.
