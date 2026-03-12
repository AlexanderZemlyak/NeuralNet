uses NumLibABC;

begin
  var v := new Vector(2,4,3);
  var m := new Matrix(3,3,1,2,3,2,4,5,9,4,1);
  Println('Исходный вектор =',v.Value);
  Println('Исходная матрица:');
  m.Value.Println;
  Println('Обратная матрица:');
  m.Inv.Value.Print;
  Println('Произведение матрицы на вектор =',(m*v).Value);
  Println('Определитель =',m.Det);
  var cond: real;
  var res := m.SLAU(v,cond);
  Println('Решение СЛАУ =',res.Value);
  Println('Число обусловленности =',cond);
end.