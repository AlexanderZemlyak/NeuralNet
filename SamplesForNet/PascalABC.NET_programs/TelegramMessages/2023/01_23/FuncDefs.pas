// 1. Полные формы записи функций.

// 1.1. Функция без параметров.
function EmptyFunc : integer;
begin
  Result := 0;
  for var i := 1 to 10 do
    Result += i;
end;

// 1.2. Функция с несколькими параметрами разных типов.
function RegularFunc(x : real; n : integer): real;
begin
  Result := 1.0;
  var Elem := 1.0;
  for var i := 1 to n do
  begin
    Elem *= x / i;
    Result += Elem;
  end;
end;

// 1.3. Функция, возвращающая несколько параметров через кортеж.
function TupleFunc(x, y: integer): (integer, integer);
begin
  Result := (x + y, x - y); // формируем кортеж
end;

// 2. Сокращенные формы записи функций.

// 2.1. Сокращенная форма записи обычной функции (с автовыводом типов).
function ShortRegularFuncAuto(a, b: real) := a + b; // возвращаем сумму, но без типа

// 2.2. Сокращенная форма записи обычной функции (без автовывода типов).
function ShortRegularFuncNoAuto(a, b: real) : real := a + b; // возвращаем сумму, но подсказываем тип

// 2.3. Сокращенная форма записи функции, возвращающей кортеж из 2 целых чисел (с автовыводом типов).
function ShortTupleFuncAuto(x, y: integer) := (x + y, x - y); // сразу возвращаем кортеж

// 2.4. Сокращенная форма записи функции, возвращающей кортеж из 2 целых чисел (без автовывода типов).
function ShortTupleFuncNoAuto(x, y: integer) : (integer, integer) := (x + y, x - y); // подсказываем тип

begin
  // 1.1.
  var sum1 := EmptyFunc; // вызов функции без параметров
  Println($'Сумма чисел от 1 до 10 = {sum1}'); Println;
  
  // 1.2.
  var (x2, n2) := (1.0, 1000); 
  var sum_exp := RegularFunc(x2, n2); // считаем экспоненту через ряд Тейлора в точке x2 с удержанием n2 слагаемых
  Println($'Точное значение функции Exp({x2}) = {Exp(x2)}');
  Println($'Приближенное значение с удержанием {n2} слагаемых = {sum_exp}');
  Println($'Модуль разности величин составляет {Abs(Exp(x2) - sum_exp)}'); Println;
  
  // 1.3.
  var (x3, y3) := (1, 2);
  var (sum3, subtract3) := TupleFunc(x3, y3); // вызов функции, возвращающей кортеж
  Println($'Сумма: {x3} + {y3} = {sum3}, {x3} - {y3} = {subtract3}'); Println;
  
  // 2.1. (2.2. аналогично)
  var (x4, y4) := (1.3, 2.2);
  var sum4 := ShortRegularFuncAuto(x4, y4); // вызов функции из п.п. 2.1
  Println($'Сумма: {x4} + {y4} = {sum4}.'); Println;
  
  // 2.3. (2.4. аналогично)
  var (x5, y5) := (10, 20);
  var (sum5, subtract5) := ShortTupleFuncAuto(x5, y5); // вызов функции из п.п. 2.3
  Println($'Сумма: {x5} + {y5} = {sum5}, {x5} - {y5} = {subtract5}');
end.