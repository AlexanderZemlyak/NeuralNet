begin
  var a := MatrRandomInteger(3,4,1,9);
  var b := MatrRandomInteger(4,5,1,9);
  a.Print;
  Println;
  b.Print;
  var c := MatrGen(a.RowCount,b.ColCount,(i,j) -> a.Row(i).Zip(b.Col(j),(x,y) -> x * y).Sum);
  Println;
  c.Print;
  
end.