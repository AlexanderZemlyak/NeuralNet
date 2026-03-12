begin
  var a := Matrrandom(3,4,1,9);
  a.Print;
  Println;
  var b := Matrrandom(4,5,1,9);
  b.Print;
  var res := MatrGen(3,5,(i,j)->a.row(i).zip(b.col(j), (x,y)->x*y).sum);
  Println;
  res.Print;
end.

