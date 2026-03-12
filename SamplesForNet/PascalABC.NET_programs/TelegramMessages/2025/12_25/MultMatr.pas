function Dot(a1, a2: array of real): real := a1.Zip(a2,(x,y) -> x*y).Sum;

function Mult(a, b: array[,] of real): array[,] of real := 
  MatrGen(a.RowCount, b.ColCount, (i,j) -> Dot(a[i,:],b[:,j]));

begin
  var a := Matr([1.1,2,3],[4.3,5,6],[7.0,8,9],[5.7,2,8]);
  var b := Matr([4.0,5],[8.2,4],[3.0,9]);
  a.Print(7,1); Println;
  b.Print(7,1); Println;
  Mult(a,b).Print(7,1);
end.